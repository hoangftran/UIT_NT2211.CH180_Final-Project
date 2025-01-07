import os
import warnings
from typing import Optional, Callable, List, Tuple

import lightgbm as lgb
from lightgbm import log_evaluation, record_evaluation, early_stopping
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import shap
from optuna.integration import LightGBMPruningCallback
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted

from scoring._utils import split_columns_by_dtype, deprecated
from scoring.metrics import gini as comp_gini

warnings.filterwarnings("ignore")


def show_plot_roc(fpr: List[List[float]], tpr: List[List[float]], gini: List[float], label: str) -> None:
    """Plots ROC curve(s) for both: CV model or single model.

    Arguments:
        fpr (list of float): list of false positive rates
        tpr (list of float): list of true positive rates
        gini (list of float): list of ginis
        label (str): label of the set ('train' or 'valid') are options used in code

    """

    plt.figure(figsize=(10, 5))

    for i in range(len(gini)):
        plt.plot(
            fpr[i],
            tpr[i],
            label="ROC curve of {} {}. fold (GINI = {})".format(label, i, round(gini[i], 3)),
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.show()


class LGBM_model(BaseEstimator):
    """LightGBM wrapper with methods suitable for credit scoring.

    Args:
        cols_pred (list of str): list of predictors, if not provided, then all columns are used
        params (dictionary): parameters of lgb.train() function
        use_CV (bool, optional): True - train n-fold CV with merged train and valid sets (default: {False})
        CV_folds (int, optional): In case of True, number of CV folds (default: {3})
        CV_seed (int, optional): In case of True, seed for k-fold split (default: {98765})

    """

    def __init__(
        self,
        cols_pred: Optional[List[str]] = None,
        params: Optional[dict] = None,
        use_CV: bool = False,
        CV_folds: int = 3,
        CV_seed: int = 98765,
    ) -> None:
        self.cols_pred = cols_pred
        self.params = params
        self.use_CV = use_CV
        self.CV_folds = CV_folds
        self.CV_seed = CV_seed
        self.explainer_ = None
        self.set_to_shap_ = None

    def __one_model(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        w_train: Optional[pd.Series] = None,
        w_valid: Optional[pd.Series] = None,
        free_raw_data: bool = False,
        verbose_level: int = 0,
    ) -> lgb.Booster:
        """Training of single lgbm model.

        Args:
            x_train (pandas.DataFrame): training set
            x_valid (pandas.DataFrame): valid set
            y_train (pandas.DataFrame): target of training set
            y_valid (pandas.DataFrame): target of valid set
            w_train (pandas.DataFrame, optional): training set weight (default: None)
            w_valid (pandas.DataFrame, optional): valid set weight (default: None)
            free_raw_data (bool, optional): free raw data after constructing inner lgbm.Dataset (default: False)
            verbose_level (int, optional): verbosity (default: 0)

        Returns:
            lgb.Booster: trained model

        """

        dtrain = lgb.Dataset(x_train, label=y_train, weight=w_train, free_raw_data=free_raw_data)
        dvalid = lgb.Dataset(x_valid, label=y_valid, weight=w_valid, reference=dtrain, free_raw_data=free_raw_data)

        evals = {}

        booster = lgb.train(
            params=self.params,
            train_set=dtrain,
            valid_sets=[dtrain, dvalid],
            valid_names=["training", "validation"],
            callbacks=[log_evaluation(verbose_level), record_evaluation(evals)],
        )

        self.evals_ = evals
        self.final_iterations_ = booster.current_iteration()

        return booster

    def show_progress(self) -> None:
        """Show curve with AUC value progress during model training iterations."""

        check_is_fitted(self, ["evals_", "final_iterations_"])

        evals = self.evals_
        final_iterations = self.final_iterations_

        print("Progress during most recent training of model.")
        plt.figure(figsize=(8, 8))
        plt.title("loss curve")
        for k, v in evals.items():
            plt.plot(np.arange(1, len(v["auc"]) + 1), v["auc"], label=k)
        plt.axvline(final_iterations, color="grey", linestyle="--")
        plt.xlabel("Iteration")
        plt.ylabel("auc")
        plt.legend(loc="lower right")
        plt.show()

        return

    def __fit_nocv(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_valid: pd.DataFrame,
        y_valid: pd.Series,
        w_train: Optional[pd.Series] = None,
        w_valid: Optional[pd.Series] = None,
        verbose_level: int = 0,
    ) -> Tuple[
        List[lgb.Booster],
        List[np.ndarray],
        List[np.ndarray],
        List[float],
        List[np.ndarray],
        List[np.ndarray],
        List[float],
    ]:
        """Train LightGBM model without cross-validation.

        Args:
            x_train (pandas.DataFrame): training set
            y_train (pandas.Series): target of training set
            x_valid (pandas.DataFrame): valid set
            y_valid (pandas.Series): target of valid set
            w_train (pandas.Series, optional): training set weight (default: None)
            w_valid (pandas.Series, optional): valid set weight (default: None)
            verbose_level (int, optional): verbosity (default: 0)

        Returns:
            List[lgbm.Booster]: list of lgbm booster models
            List[numpy.ndarray]: list of fpr values for training set
            List[numpy.ndarray]: list of tpr values for training set
            List[float]: list of gini values for training set
            List[numpy.ndarray]: list of fpr values for valid set
            List[numpy.ndarray]: list of tpr values for valid set
            List[float]: list of gini values for valid set

        """
        print("I am not using CV option, training is stopped when model starts to overfit on valid set")
        print(" ")

        model = self.__one_model(x_train, y_train, x_valid, y_valid, w_train, w_valid, verbose_level=verbose_level)
#         fpr_train, tpr_train, _ = roc_curve(y_train, model.predict(x_train), sample_weight=w_train)
#         gini_train = comp_gini(y_train, model.predict(x_train), sample_weight=w_train)
#         fpr_valid, tpr_valid, _ = roc_curve(y_valid, model.predict(x_valid), sample_weight=w_valid)
#         gini_valid = comp_gini(y_valid, model.predict(x_valid), sample_weight=w_valid)

#         fpr_trains = [fpr_train]
#         tpr_trains = [tpr_train]
#         gini_trains = [gini_train]

#         fpr_valids = [fpr_valid]
#         tpr_valids = [tpr_valid]
#         gini_valids = [gini_valid]

        models = [model]
#         print(len(fpr_trains[0]))
#         return models, fpr_trains, tpr_trains, gini_trains, fpr_valids, tpr_valids, gini_valids
        return models

    def __fit_cv(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_valid: pd.DataFrame,
        y_valid: pd.Series,
        w_train: pd.Series = None,
        w_valid: pd.Series = None,
        verbose_level: int = 0,
    ) -> Tuple[
        List[lgb.Booster],
        List[np.ndarray],
        List[np.ndarray],
        List[float],
        List[np.ndarray],
        List[np.ndarray],
        List[float],
    ]:
        """Training of lgbm model with cross-validation.

        Args:
            x_train (pandas.DataFrame): training set
            y_train (pandas.Series): target of training set
            x_valid (pandas.DataFrame): valid set
            y_valid (pandas.Series): target of valid set
            w_train (pandas.Series, optional): training set weight (default: {None})
            w_valid (pandas.Series, optional): valid set weight (default: {None})
            verbose_level (int, optional): verbosity (default: 0)

        Returns:
            models (list): list of lgbm booster models
            fpr_train (list): list of false positive rates for training set
            tpr_train (list): list of true positive rates for training set
            gini_train (list): list of gini values for training set
            fpr_valid (list): list of false positive rates for valid set
            tpr_valid (list): list of true positive rates for valid set
            gini_valid (list): list of gini values for valid set

        """
        print("I am using CV option, train and valid (if provided) sets are merged")
        print(" ")

        merged_ds = x_train
        merged_target = y_train
        if (x_valid is not None) and (y_valid is not None):
            merged_ds = pd.concat([x_train, x_valid]).reset_index(drop=True)
            merged_target = pd.concat([y_train, y_valid]).reset_index(drop=True)
        merged_ds = merged_ds.reset_index(drop=True)
        merged_target = merged_target.reset_index(drop=True)

        merged_weight = None
        if (w_train is not None) and (w_valid is not None):
            merged_weight = pd.concat([w_train, w_valid]).reset_index(drop=True)

        folds = StratifiedKFold(n_splits=self.CV_folds, shuffle=True, random_state=self.CV_seed)

        models = []
        fpr_train, fpr_valid = [], []
        tpr_train, tpr_valid = [], []
        gini_train, gini_valid = [], []

        for train_index, valid_index in folds.split(merged_ds, merged_target):
            X_train, X_valid = (
                merged_ds.loc[train_index, :],
                merged_ds.loc[valid_index, :],
            )
            Y_train, Y_valid = (
                merged_target.loc[train_index],
                merged_target.loc[valid_index],
            )
            W_train, W_valid = None, None
            if merged_weight is not None:
                W_train, W_valid = (
                    merged_weight.loc[train_index],
                    merged_weight.loc[valid_index],
                )

            model = self.__one_model(
                X_train, Y_train, X_valid, Y_valid, w_train=W_train, w_valid=W_valid, verbose_level=verbose_level
            )
            models.append(model)
#             fpr, tpr, _ = roc_curve(Y_train, model.predict(X_train), sample_weight=W_train)
#             fpr_train.append(fpr)
#             tpr_train.append(tpr)
#             gini_train.append(comp_gini(Y_train, model.predict(X_train), sample_weight=W_train))

#             fpr, tpr, _ = roc_curve(Y_valid, model.predict(X_valid), sample_weight=W_valid)
#             fpr_valid.append(fpr)
#             tpr_valid.append(tpr)
#             gini_valid.append(comp_gini(Y_valid, model.predict(X_valid), sample_weight=W_valid))

#         return models, fpr_train, tpr_train, gini_train, fpr_valid, tpr_valid, gini_valid
        return models

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        w_train: Optional[pd.Series] = None,
        w_valid: Optional[pd.Series] = None,
        show_plots: bool = True,
        verbose_level: int = 0,
    ) -> List[lgb.Booster]:
        """Fitting of a model IF use_CV=False Fits model from train set and
        training is stopped when it starts to overfit on valid set.

            IF use_CV=True
                Merges train and valid sets and fits n-time CV


        Args:
            X_train (pandas.DataFrame): training set
            X_valid (pandas.DataFrame): valid set
            y_train (pandas.DataFrame): target of training set
            y_valid (pandas.DataFrame): target of valid set
            w_train (pandas.DataFrame, optional): training set weight (default: None)
            w_valid (pandas.DataFrame, optional): valid set weight (default: None)
            show_plots (bool, optional): show plots (default: True)
            verbose_level (int, optional): verbosity (default: 0)

        Returns:
            list of lgbm.Booster:
                IF use_CV = False
                    list with one lgb.booster model
                IF use_CV = True
                    list with n lgb.booster models

        """
        if not self.use_CV:
            self.CV_folds = 1

        if self.cols_pred:
            self.feature_names_in_ = self.cols_pred
            X_train = X_train[self.feature_names_in_]
            if X_valid is not None:
                X_valid = X_valid[self.feature_names_in_]
            self.n_features_in_ = len(self.feature_names_in_)

        else:
            super()._check_n_features(X_train, reset=True)
            super()._check_feature_names(X_train, reset=True)

        _, x_cols_cat = split_columns_by_dtype(X_train)
        self.cols_cat_ = list(set(self.feature_names_in_).intersection(x_cols_cat))

        if not self.use_CV and (X_valid is None or y_valid is None):
            raise ValueError("Validation set is required to train not-CV model.")

        if not self.use_CV:
            models = self.__fit_nocv(
                X_train, y_train, X_valid, y_valid, w_train, w_valid, verbose_level=verbose_level
            )

        else:
            models = self.__fit_cv(
                X_train, y_train, X_valid, y_valid, w_train, w_valid, verbose_level=verbose_level
            )
#         if show_plots:
#             show_plot_roc(fpr_train, tpr_train, gini_train, "train")
#             show_plot_roc(fpr_valid, tpr_valid, gini_valid, "valid")

        self.models_ = models

        return self.models_

    def score(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None) -> float:
        """Computes gini score on validation set.

        Args:
            X (pandas.DataFrame): validation set
            y (pandas.Series): target of validation set
            sample_weight (pandas.Series, optional): validation set weight (default: None)

        Returns:
            float: gini score

        """
        return comp_gini(y, self.predict(X), sample_weight=sample_weight)

    def _comp_var_imp(self, models: List[lgb.Booster] = None) -> pd.DataFrame:
        """Creates dataframe with variable importances. In case of use_CV = True than compute average var. imp. based on all models

        Args:
            models (list of lgbm.Booster, optional): (default: None)

        Returns:
            pandas.DataFrame: df with gain and weight importance for all features
        """

        if (models is None) and (not hasattr(self, "models_")):
            raise NotFittedError("Model was not trained yet.")
        elif models is None:
            models = self.models_

        importance_df = pd.DataFrame()
        importance_df["Feature"] = self.feature_names_in_
        importance_df["importance_gain"] = 0
        importance_df["importance_weight"] = 0

        for model in models:
            importance_df["importance_gain"] = (
                importance_df["importance_gain"] + model.feature_importance(importance_type="gain") / self.CV_folds
            )
            importance_df["importance_weight"] = (
                importance_df["importance_weight"] + model.feature_importance(importance_type="split") / self.CV_folds
            )

        return importance_df

    def plot_imp(
        self,
        models: Optional[List[lgb.Booster]] = None,
        imp_type: str = "importance_gain",
        n_predictors: int = 100,
        show_plots: bool = True,
        output_folder: Optional[str] = None,
    ) -> pd.DataFrame:
        """Takes output of _comp_var_imp and print top n predictors in nice form,
        sorted by highest importance.

        Args:
            models (list of lgbm.Booster, optional): list of lgbm.boosters
            imp_type (string, optional): 'importance_gain' or 'importance_weight'
            n_predictors (int, optional): number of best features (default: {100})
            show_plots (bool, optional): show plots (default: {True})
            output_folder (string, optional): path to output folder (default: {None})

        Raises:
            ValueError: if 'imp_type' parameter is different from 'importance_gain' or 'importance_weight'

        Returns:
            pandas.Dataframe: df with features a importances

        """

        if (models is None) and (not hasattr(self, "models_")):
            raise NotFittedError("Model was not trained yet.")
        elif models is None:
            models = self.models_

        if (imp_type != "importance_gain") & (imp_type != "importance_weight"):
            raise ValueError('Only "importance_gain" and "importance_weight" are possible imp_types.')

        df: pd.DataFrame = self._comp_var_imp(models)

        if show_plots or (output_folder is not None):
            plt.figure(figsize=(20, n_predictors / 2))
            sns.barplot(
                x=imp_type,
                y="Feature",
                data=df.sort_values(by=imp_type, ascending=False).head(n_predictors),
            )
            if output_folder is not None:
                plt.savefig(output_folder + "/feat_importance.png", bbox_inches="tight")
                plt.close()
            if show_plots:
                plt.show()

        return df.sort_values(by=imp_type, ascending=False).head(n_predictors)[["Feature", imp_type]]

    def predict(self, X: pd.DataFrame, models: Optional[List[lgb.Booster]] = None) -> np.ndarray:
        """Predicts probabilities on new dataset.

        Args:
            X (pandas.DataFrame): set for which you want to predict output
            models (list of lgbm.Booster, optional): list of lgbm.boosters

        Returns:
            numpy.array: predicted class labels for multiclass, probabilities for binary
        """
        if (models is None) and (not hasattr(self, "models_")):
            raise NotFittedError("Model was not trained yet and the parameter 'models' was not provided.")
        elif models is None:
            models = self.models_

        # Get predictions from first model to determine output shape
        first_pred = models[0].predict(X[self.feature_names_in_])

        # Check if multiclass (output will be 2D array)
        is_multiclass = len(first_pred.shape) > 1

        if is_multiclass:
            # For multiclass, sum probabilities for each class
            predictions = np.zeros(first_pred.shape)
            for model in models:
                predictions += model.predict(X[self.feature_names_in_])
            predictions /= self.CV_folds
            # Return class with highest probability
            return np.argmax(predictions, axis=1)
        else:
            # For binary classification, average probabilities
            predictions = np.zeros(X.shape[0])
            for model in models:
                predictions += model.predict(X[self.feature_names_in_])
            predictions /= self.CV_folds
            return predictions

    def marginal_contribution(
        self,
        x_train: pd.DataFrame,
        x_valid: pd.DataFrame,
        y_train: pd.Series,
        y_valid: pd.Series,
        set_to_test: pd.DataFrame,
        set_to_test_target: pd.Series,
        w_train: Optional[pd.Series] = None,
        w_valid: Optional[pd.Series] = None,
        set_to_test_weight: Optional[pd.Series] = None,
        silent: Optional[bool] = False,
        verbose_level: int = 0,
    ) -> pd.DataFrame:
        """Computes gini performance of model on set_to_test, trained without
        particular feature. This is computed for every feature separately from
        self.feature_names_in_.

        Args:
            x_train (pandas.DataFrame): training set
            x_valid (pandas.DataFrame):valid set
            y_train (pandas.DataFrame): target of training set
            y_valid (pandas.DataFrame): target of valid set
            set_to_test (pandas.DataFrame): set on which you want to compute marginal contribution
            set_to_test_target (pandas.DataFrame): target of set on which you want to compute marginal contribution
            w_train (pandas.DataFrame, optional):training set weights (default: None)
            w_valid (pandas.DataFrame, optional): valid set weights (default: None)
            set_to_test_weight (pandas.DataFrame, optional): set on which you want to compute marginal contribution weights (default: None)
            silent (boolean, optional): whether the output should NOT be displayed on screen (default: False)
            verbose_level (int, optional): verbosity (default: 0)

        Returns:
            pandas.DataFrame: dataframe with 4 columns - feature, gini with feature, gini without feature and difference of gini with feature and gini without feature

        """

        diff_dataframe = pd.DataFrame(columns=["Feature", "Perf_with", "Perf_without", "Difference"])
        diff_dataframe = diff_dataframe.fillna(0)

        predictors = self.feature_names_in_

        model = self.__one_model(
            x_train[predictors],
            y_train,
            x_valid[predictors],
            y_valid,
            w_train=w_train,
            w_valid=w_valid,
            verbose_level=verbose_level,
        )

        pred_set_to_test = model.predict(set_to_test[predictors])
        gini_test_original = comp_gini(set_to_test_target, pred_set_to_test, sample_weight=set_to_test_weight)

        j = 0
        for i, pred in enumerate(predictors):
            predictors_new = list(predictors.copy())
            predictors_new.remove(pred)

            if "monotone_constraints" in self.params:
                monotone_con_original = self.params["monotone_constraints"]
                self.params["monotone_constraints"] = (
                    self.params["monotone_constraints"][:i] + self.params["monotone_constraints"][i + 1 :]
                )

            model = self.__one_model(
                x_train[predictors_new],
                y_train,
                x_valid[predictors_new],
                y_valid,
                w_train=w_train,
                w_valid=w_valid,
                verbose_level=verbose_level,
            )
            pred_set_to_test = model.predict(set_to_test[predictors_new])

            if "monotone_constraints" in self.params:
                self.params["monotone_constraints"] = monotone_con_original

            diff_dataframe.loc[j, "Feature"] = pred
            diff_dataframe.loc[j, "Perf_with"] = gini_test_original * 100
            diff_dataframe.loc[j, "Perf_without"] = (
                comp_gini(
                    set_to_test_target,
                    pred_set_to_test,
                    sample_weight=set_to_test_weight,
                )
                * 100
            )
            diff_dataframe.loc[j, "Difference"] = (
                diff_dataframe.loc[j, "Perf_with"] - diff_dataframe.loc[j, "Perf_without"]
            )

            j += 1

        if not silent:
            print(diff_dataframe.sort_values(by=["Difference"]).to_string())

        return diff_dataframe.sort_values(by=["Difference"])

    @deprecated(
        reason=(
            "The use of numerical/categorical columns as parameter will be deprecated in future, "
            "as the dtypes are now automatically inferred from the data. "
            "Please use 'print_shap_values_new' without passing the columns as arguments."
        )
    )
    def print_shap_values_old(
        self,
        cols: List[str],
        cols_cat: List[str],
        x_train: pd.DataFrame,
        x_valid: pd.DataFrame,
        y_train: pd.Series,
        y_valid: pd.Series,
        set_to_shap: pd.DataFrame,
        w_train: Optional[pd.Series] = None,
        w_valid: Optional[pd.Series] = None,
        output_folder: Optional[str] = None,
    ) -> pd.DataFrame:
        """This method computes shap values for given set_to_shap. Deprecated.
        Please use 'print_shap_values_new' instead.

        Args:
            cols (List[str]): numerical columns
            cols_cat (List[str]): categorical columns
            x_train (pandas.DataFrame): training set
            x_valid (pandas.DataFrame): valid set
            y_train (pandas.DataFrame): target of training set
            y_valid (pandas.DataFrame): target of valid set
            set_to_shap (pandas.DataFrame): set on which you want to compute shap values
            output_folder (str): folder to output charts
            w_train (pandas.DataFrame, optional): training set weights (default: None)
            w_valid (pandas.DataFrame, optional): valid set weights (default: None)

        Returns:
            pandas.DataFrame: dataframe with shap values

        """

        return self.print_shap_values(
            x_train,
            x_valid,
            y_train,
            y_valid,
            set_to_shap,
            w_train=w_train,
            w_valid=w_valid,
            output_folder=output_folder,
        )

    def print_shap_values(
        self,
        x_train: pd.DataFrame,
        x_valid: pd.DataFrame,
        y_train: pd.Series,
        y_valid: pd.Series,
        set_to_shap: pd.DataFrame,
        w_train: Optional[pd.Series] = None,
        w_valid: Optional[pd.Series] = None,
        output_folder: Optional[str] = None,
        show_plots: bool = True,
        verbose_level: int = 0,
        max_display: int = 20,
    ) -> pd.DataFrame:
        """This method computes shap values for given set_to_shap.

        Args:
            x_train (pandas.DataFrame): training set
            x_valid (pandas.DataFrame): valid set
            y_train (pandas.DataFrame): target of training set
            y_valid (pandas.DataFrame): target of valid set
            set_to_shap (pandas.DataFrame): set on which you want to compute shap values
            output_folder (str): folder to output charts
            w_train (pandas.DataFrame, optional): training set weights (default: None)
            w_valid (pandas.DataFrame, optional): valid set weights (default: None)
            output_folder (str, optional): where the pictures should be saved. If None, they won't be saved (default: None)
            show_plots (bool, optional): whether to show the plots or not (default: True)
            verbose_level (int, optional): verbosity (default: 0)
            max_display (int, optional): maximum number of features to display (default: 20)

        Returns:
            pandas.DataFrame: Dataframe with name of the feature and its shap value

        """
        warnings.warn(
            (
                "Method 'print_shap_values' does not use the 'cols' and 'cols_cat' parameters anymore. If you want to"
                " use the old behaviour, please use 'print_shap_values_old' instead."
            ),
            DeprecationWarning,
        )
        print("Model has to be trained again because of categorical variables encoding")
        print(" ")

        # if the model has not been trained before we have to find out what the columns and cat_columns are
        if not hasattr(self, "feature_names_in_"):
            self._set_features_in(x_train)

        train = x_train.copy()
        x_train = x_train[self.feature_names_in_]
        x_valid = x_valid[self.feature_names_in_]
        set_to_shap = set_to_shap[self.feature_names_in_]

        for col in self.cols_cat_:
            x_train[col] = x_train[col].cat.add_categories("NA").fillna("NA")
            x_valid[col] = x_valid[col].cat.add_categories("NA").fillna("NA")
            set_to_shap[col] = set_to_shap[col].cat.add_categories("NA").fillna("NA")
            _, indexer = pd.factorize(x_train[col])
            x_train[col] = indexer.get_indexer(x_train[col])
            x_valid[col] = indexer.get_indexer(x_valid[col])
            set_to_shap[col] = indexer.get_indexer(set_to_shap[col])

        model = self.__one_model(
            x_train, y_train, x_valid, y_valid, w_train=w_train, w_valid=w_valid, verbose_level=verbose_level
        )

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(set_to_shap)

        self.explainer_ = explainer
        self.set_to_shap_ = set_to_shap
        self.train_ = train

        if isinstance(shap_values, list):
            # this if - else covers problem discussed here
            # https://github.com/slundberg/shap/issues/526
            # explainer.shap_values() behaves differently on a local laptop and on server

            self.shap_values_ = shap_values[1]
            self.base_value_ = explainer.expected_value[1]

        else:
            self.shap_values_ = shap_values
            self.base_value_ = explainer.expected_value

        if output_folder is not None and not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if show_plots or output_folder is not None:
            shap.summary_plot(self.shap_values_, set_to_shap, show=show_plots, max_display=max_display)
            if output_folder is not None:
                plt.savefig(output_folder + "/shap.png", bbox_inches="tight")
                plt.close()

            shap.summary_plot(self.shap_values_, set_to_shap, plot_type="bar", show=show_plots, max_display=max_display)
            if output_folder is not None:
                plt.savefig(output_folder + "/shap_abs.png", bbox_inches="tight")
                plt.close()

        var_imp_dataframe = {
            "Feature": self.feature_names_in_,
            "Shap_importance": np.mean(abs(self.shap_values_), axis=0),
        }

        return pd.DataFrame(var_imp_dataframe).sort_values(by=["Shap_importance"], ascending=False)

    def print_shap_interaction_matrix(
        self,
        output_folder: Optional[str] = None,
        show_plots: bool = True,
    ) -> None:
        """Prints shap interaction matrix, based on
        https://christophm.github.io/interpretable-ml-book/shap.html#shap-
        interaction-values It prints sum of absolute interactions values throught
        all observations. Diagonal values are manually set to zero.

        Args:
            output_folder (str, optional): where the pictures should be saved. If None, they won't be saved (default: None)
            show_plots (bool, optional): whether to show the plots or not (default: True)

        Returns:
            None

        """
        if self.explainer_ is None or self.set_to_shap_ is None:
            raise ValueError("You have to compute shap values first. Use 'print_shap_values()'.")

        shap_inter_values = self.explainer_.shap_interaction_values(self.set_to_shap_)

        corr = np.sum(abs(shap_inter_values), axis=0)
        corr[np.diag_indices_from(corr)] = 0

        corr = pd.DataFrame(corr, columns=self.feature_names_in_, index=self.feature_names_in_)
        corr = round(corr, 0)

        if output_folder is not None and not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if show_plots or output_folder is not None:
            sns.set(rc={"figure.figsize": (20, 20)})
            sns.heatmap(
                corr,
                annot=True,
                xticklabels=self.feature_names_in_,
                yticklabels=self.feature_names_in_,
                fmt=".5g",
            )
            if output_folder is not None:
                plt.savefig(output_folder + "/shap_int_matrix.png", bbox_inches="tight")
                plt.close()
            if show_plots:
                plt.show()

        return corr

    def shap_one_row(self, row: int) -> None:
        """Prints decision plot for 1 row.

        Args:
            row (int): row number of self.set_to_shap

        """

        shap.force_plot(
            self.base_value_,
            self.shap_values_[row, :],
            self.set_to_shap_.iloc[row, :],
            link="identity",
            matplotlib=True,
        )

        shap.force_plot(
            self.base_value_,
            self.shap_values_[row, :],
            self.set_to_shap_.iloc[row, :],
            link="logit",
            matplotlib=True,
        )

    def shap_dependence_plot(
        self,
        x: str,
        y: Optional[str] = None,
        xmin: Optional[float] = None,
        xmax: Optional[float] = None,
        output_folder: Optional[str] = None,
        show_plots: bool = True,
    ) -> None:
        """Prints shap dependence plot for given feature. If y is not specified,
        algorithm finds it automatically.

        Arguments:
            x {string} -- feature name

        Keyword Arguments:
            y {string} -- feature name (default: {None})
            xmin {float} -- min value on x axis to be displayed (default: {None})
            xmax {float} -- max value on x axis to be displayed (default: {None})

        """

        if x in self.cols_cat_:
            print("Encoding of categories for your variable")
            labels, uniques = pd.factorize(self.train_[x].cat.add_categories("NA").fillna("NA"))
            for j in range(len(np.unique(labels))):
                print(np.unique(labels)[j])
                print(uniques[j])

        if show_plots or output_folder is not None:
            if y is None:
                shap.dependence_plot(x, self.shap_values_, self.set_to_shap_, xmin=xmin, xmax=xmax, show=show_plots)
            else:
                shap.dependence_plot(
                    x, self.shap_values_, self.set_to_shap_, interaction_index=y, xmin=xmin, xmax=xmax, show=show_plots
                )
            if output_folder is not None:
                plt.savefig(output_folder + "/shap_dependence_plot.png", bbox_inches="tight")
                plt.close()

    def param_hyperopt_objective(
        self,
        trial: optuna.Trial,
        space_callable: Optional[Callable] = None,
        lgbm_random_state: Optional[int] = None,
        CV_folds: Optional[int] = None,
        CV_seed: Optional[int] = None,
        merged_ds: Optional[pd.DataFrame] = None,
        merged_target: Optional[pd.Series] = None,
        merged_weight: Optional[pd.Series] = None,
        n_jobs: Optional[int] = None,
    ) -> float:
        """Objective function for hyperoptimization.

        Args:
            trial (optuna.Trial): trial object
            space_callable (function, optional): function that returns dictionary with hyperparameters (default: None)
            lgbm_random_state (int, optional): random state for lgbm (default: None)
            CV_folds (int, optional): number of folds for cross validation (default: None)
            CV_seed (int, optional): seed for cross validation (default: None)
            merged_ds (pd.DataFrame, optional): dataset (default: None)
            merged_target (pd.Series, optional): target (default: None)
            merged_weight (pd.Series, optional): weight (default: None)
            n_jobs (int, optional): number of jobs (default: None)

        Returns:
            float: score

        """
        space_user = {}
        if space_callable is not None:
            space_user = space_callable(trial)

        space_default = {
            "n_estimators": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 2, 64),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 500, step=5),
            "min_child_weight": trial.suggest_int("min_child_weight", 10, 100, step=10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 0.95, step=0.1),
            "bagging_freq": 1,
            "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 0.95, step=0.1),
            "objective": "binary",
            "metric": "auc",
            "seed": lgbm_random_state,
            "verbosity": -1,
            "nthreads": -1,
            "boosting_type": "gbdt",
        }
        params = {}
        for key in space_default:
            if key not in space_user:
                params[key] = space_default[key]
        for key in space_user:
            if key not in params:
                params[key] = space_user[key]

        skfold = StratifiedKFold(n_splits=CV_folds, shuffle=True, random_state=CV_seed)

        cv_scores = []
        merged_ds.reset_index(drop=True, inplace=True)
        merged_target.reset_index(drop=True, inplace=True)
        for idx, (train_idx, val_idx) in enumerate(skfold.split(merged_ds, merged_target)):
            X_train_, y_train_, w_train_ = (
                merged_ds.iloc[train_idx],
                merged_target.iloc[train_idx],
                merged_weight.loc[train_idx],
            )
            X_val_, y_val_, w_val_ = (
                merged_ds.iloc[val_idx],
                merged_target.iloc[val_idx],
                merged_weight.loc[val_idx],
            )

            model = lgb.LGBMClassifier(n_jobs=n_jobs, **params)
            model.fit(
                X_train_,
                y_train_,
                sample_weight=w_train_,
                eval_set=[(X_val_, y_val_)],
                eval_sample_weight=[w_val_],
                callbacks=[
                    LightGBMPruningCallback(trial, "auc"),
                    early_stopping(100, first_metric_only=True),
                ],  # Add a pruning callback
            )

            val_pred = model.predict_proba(X_val_)[:, 1]
            val_gini = 2 * roc_auc_score(y_val_, val_pred) - 1
            cv_scores.append(val_gini)

        avg_val_gini = float(np.mean(cv_scores))
        print("Actual average gini:")
        print(avg_val_gini)
        print("----------------------------------------------------------------------------------")

        return avg_val_gini

    def param_hyperopt(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_valid: pd.DataFrame = None,
        y_valid: pd.Series = None,
        w_train: pd.Series = None,
        w_valid: pd.Series = None,
        CV_folds: int = 5,
        n_iter: int = 500,
        n_jobs: int = -1,
        space_callable: Optional[Callable[[optuna.Trial], dict]] = None,
        n_startup_trials: int = 32,
        lgbm_random_state: int = 42,
        CV_seed: int = 42,
        timeout: int = 3600,
        show_progress_bar: bool = False,
    ) -> Tuple[dict, optuna.Study]:
        """Finds optimal hyperparameters based on cross validation AUC.

        Args:
            x_train (pandas.DataFrame): training set
            x_valid (pandas.DataFrame): valid set
            y_train (pandas.DataFrame): target of training set
            y_valid (pandas.DataFrame): target of valid set
            w_train (pandas.DataFrame, optional): training set weights (default: None)
            w_valid (pandas.DataFrame, optional): valid set weights (default: None)
            n_iter (int, optional): number of iteration (default: {500})
            n_jobs (int, optional): number of jobs for LGBM (default: {-1}, that means all available cores)
            space_callable (Callable[[optuna.Trial], dict], optional): callable function to define hyperparameter space (default: {None})
            n_startup_trials (int, optional): number of startup trials (default: {32})
            lgbm_random_state (int, optional): random state for lgbm (default: {42})
            CV_seed (int, optional): random state for cross validation (default: {42})
            timeout (int, optional): timeout for hyperparameter optimization (default: {3600})
            show_progress_bar (bool, optional): show progress bar (default: {False})

        Returns:
            dict: optimal hyperparameters
            study: optuna Study object

        """

        if not hasattr(self, "feature_names_in_"):
            self._set_features_in(x_train)

        if x_valid is None or y_valid is None:
            merged_ds = x_train[self.feature_names_in_].reset_index(drop=True)
            merged_target = y_train.reset_index(drop=True)
            merged_weight = w_train if w_train else pd.DataFrame([1] * (len(merged_target)))

        else:
            merged_ds = pd.concat([x_train[self.feature_names_in_], x_valid[self.feature_names_in_]]).reset_index(
                drop=True
            )
            merged_target = pd.concat([y_train, y_valid]).reset_index(drop=True)
            if (w_train is not None) and (w_valid is not None):
                merged_weight = pd.concat([w_train, w_valid]).reset_index(drop=True)
            else:
                merged_weight = pd.DataFrame([1] * (len(merged_target)))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(n_startup_trials=n_startup_trials),
        )
        study.optimize(
            lambda trial: self.param_hyperopt_objective(
                trial,
                space_callable=space_callable,
                lgbm_random_state=lgbm_random_state,
                CV_folds=CV_folds,
                CV_seed=CV_seed,
                merged_ds=merged_ds,
                merged_target=merged_target,
                merged_weight=merged_weight,
                n_jobs=n_jobs,
            ),
            n_trials=n_iter,
            timeout=timeout,
            show_progress_bar=show_progress_bar,
        )
        best_params = study.best_params

        print("Best combination of parameters is:")
        print(best_params)
        print("")
        return best_params, study

    def _set_features_in(self, x_train: pd.DataFrame) -> None:
        """Sets the feature names in the model.

        Args:
            x_train (pandas.DataFrame): training set

        """
        if not self.cols_pred:
            self.feature_names_in_ = x_train.columns
        else:
            self.feature_names_in_ = self.cols_pred

        _, cols_cat = split_columns_by_dtype(x_train[self.feature_names_in_])
        self.cols_cat_ = cols_cat
