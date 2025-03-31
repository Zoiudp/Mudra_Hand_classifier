import torch
import joblib
import pandas as pd
import numpy as np
from Autoencoder import Autoencoder # Assuming Autoencoder.py contains the Autoencoder class

def load_models():
    # Load the trained models and scaler
    try:
        # Load the autoencoder model
        autoencoder = Autoencoder(input_dim=24, latent_dim=10)
        autoencoder.load_state_dict(torch.load('mudra_autoencoder_model.pth'))
        autoencoder.eval()  # Set to evaluation mode
        
        # Load the Random Forest classifier
        clf = joblib.load('mudra_random_forest_classifier.joblib')
        
        # Load the scaler
        scaler = joblib.load('mudra_scaler.joblib')
        
        return autoencoder, clf, scaler
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None

def predict_mudra(data, autoencoder=None, clf=None, scaler=None):
    """
    Predict the Mudra class for a single data point
    
    Parameters:
    data: pandas.Series or numpy.ndarray - the input data point
    autoencoder, clf, scaler: pre-loaded models (optional)
    
    Returns:
    predicted_class: str - the predicted Mudra class
    """
    # Load models if not provided
    if autoencoder is None or clf is None or scaler is None:
        autoencoder, clf, scaler = load_models()
    
    if isinstance(data, pd.DataFrame):
        # If a DataFrame with multiple rows is provided, use only numeric columns
        numeric_data = data.select_dtypes(include=['number'])
        # Exclude the target column and time column
        if 'Mudra' in numeric_data.columns:
            numeric_data = numeric_data.drop(columns=['Mudra'])
        if 'Time' in numeric_data.columns:
            numeric_data = numeric_data.drop(columns=['Time'])
    elif isinstance(data, pd.Series):
        # If a Series is provided, convert to DataFrame and drop non-numeric values
        numeric_data = pd.DataFrame([data.values], columns=data.index)
        numeric_data = numeric_data.select_dtypes(include=['number'])
        # Exclude the target column and time column
        if 'Mudra' in numeric_data.columns:
            numeric_data = numeric_data.drop(columns=['Mudra'])
        if 'Time' in numeric_data.columns:
            numeric_data = numeric_data.drop(columns=['Time'])
    else:
        # If numpy array, convert to DataFrame
        numeric_data = pd.DataFrame([data])
    
    # Scale the data
    scaled_data = scaler.transform(numeric_data)
    
    # Convert to PyTorch tensor
    tensor_data = torch.tensor(scaled_data, dtype=torch.float32)
    
    # Get latent representation
    with torch.no_grad():
        latent_repr = autoencoder.encoder(tensor_data).numpy()
    
    # Make prediction
    prediction = clf.predict(latent_repr)[0]
    
    return prediction

def test_with_sample_from_csv(csv_path, num_tests=10000):
    """
    Test the prediction function with multiple random samples from a CSV file
    
    Parameters:
    csv_path: str - path to the CSV file
    num_tests: int - number of random predictions to make
    
    Returns:
    success_count: int - number of correct predictions
    fail_count: int - number of incorrect predictions
    success_rate: float - percentage of correct predictions
    """
    try:
        # Load models
        autoencoder, clf, scaler = load_models()
        
        # Load the CSV file
        df = pd.read_csv(csv_path)
        #df = df.dropna(subset=['Mudra'])  # Drop rows with NaN values
        
        # Initialize counters
        success_count = 0
        fail_count = 0
        
        # Perform multiple random predictions
        for i in range(num_tests):
            # Get a random row
            random_index = np.random.randint(0, len(df))
            sample = df.iloc[random_index]
            
            # Make prediction
            prediction = predict_mudra(sample, autoencoder, clf, scaler)
            
            # Get actual class if available
            actual_class = sample.get('Mudra', "Unknown")
            
            # Check if prediction is correct
            if prediction == actual_class:
                success_count += 1
                result = "✓ Correct"
            else:
                fail_count += 1
                result = "✗ Wrong"
            
            # Print results
            print(f"Test {i+1} (Index {random_index}):")
            print(f"  Predicted: {prediction}")
            print(f"  Actual: {actual_class}")
            print(f"  Result: {result}")
        
        # Calculate success rate
        success_rate = (success_count / num_tests) * 100 if num_tests > 0 else 0
        
        # Print summary
        print(f"\nSummary:")
        print(f"  Success: {success_count} points")
        print(f"  Failures: {fail_count} points")
        print(f"  Success Rate: {success_rate:.2f}%")
        
        return success_count, fail_count, success_rate
    except Exception as e:
        print(f"Error during prediction: {e}")
        return 0, 0, 0
# Example usage
if __name__ == "__main__":
    # Load a sample from the combined data CSV
    test_with_sample_from_csv('combined_data_with_classification.csv')