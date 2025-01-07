import pandas as pd
import numpy as np
import pickle

def load_model_components():
    """Load the saved model and components"""
    with open('model/lightgbm_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('model/features.pkl', 'rb') as f:
        features = pickle.load(f)
    
    with open('model/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    return model, features, label_encoder

def predict(data_path):
    """
    Make predictions on new data
    
    Args:
        data_path: Path to CSV file containing data to predict
    
    Returns:
        DataFrame with original data and predictions
    """
    # Load model components
    model, features, label_encoder = load_model_components()
    
    # Read and prepare data
    data = pd.read_csv(data_path)
    X = data[features]
    
    # Make predictions
    predictions_numeric = model.predict(X)
    
    # Convert predictions to original classes
    predictions = label_encoder.inverse_transform(predictions_numeric)
    
    # Add predictions to dataframe
    data['Label'] = predictions
    
    return data

if __name__ == "__main__":
    data_path = "path/to/your/test_data.csv"
    predictions = predict(data_path)
    predictions.to_csv('predictions.csv')