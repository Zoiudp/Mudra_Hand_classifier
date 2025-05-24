import torch
import joblib
import pandas as pd
import numpy as np
from Autoencoder import Autoencoder # Assuming Autoencoder.py contains the Autoencoder class
import os
# Carregar o modelo treinado, scaler e label_encoder
from tensorflow.keras.models import load_model
import pickle


def load_models():
    # Load the trained models and scaler
    try:
        # Defina o caminho para o diretório onde o arquivo está localizado
        diretorio = 'Hand_classifier'

        # Construa o caminho completo para o arquivo .pth
        caminho_autoencoder = os.path.join(diretorio, 'mudra_autoencoder_model.pth')
        caminho_random_forest = os.path.join(diretorio, 'mudra_random_forest.joblib')
        caminho_scaler = os.path.join(diretorio, 'mudra_scaler.joblib')
        caminho_pca = os.path.join(diretorio, 'pca_model.pkl')

        # Load the autoencoder model
        autoencoder = Autoencoder(input_dim=12, latent_dim=10)
        autoencoder.load_state_dict(torch.load(caminho_autoencoder))
        autoencoder.eval()  # Set to evaluation mode
        
        # Load the Random Forest classifier
        clf = joblib.load(caminho_random_forest)
        
        # Load the scaler
        scaler = joblib.load(caminho_scaler)

        # Load the t-SNE model
        pca = joblib.load(caminho_pca)
        # Check if models are loaded correctly
        if autoencoder is None or clf is None or scaler is None or pca is None:
            raise ValueError("One or more models failed to load.")
        print("Models loaded successfully.")
        # Return the loaded models
        return autoencoder, clf, scaler, pca
    
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None

def predict_mudra(data, autoencoder=None, clf=None, scaler=None, pca=None):
    """
    Predict the Mudra class for a single data point
    
    Parameters:
    data: pandas.Series or numpy.ndarray - the input data point
    autoencoder, clf, scaler: pre-loaded models (optional)
    
    Returns:
    predicted_class: str - the predicted Mudra class
    """
    # Load models if not provided
    if autoencoder is None or clf is None or scaler is None or pca is None:
        autoencoder, clf, scaler, pca = load_models()
    
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
    
    latent_repr_tsne = pca.transform(latent_repr)
    # Make prediction
    prediction = clf.predict(latent_repr_tsne)[0]
    
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
        autoencoder, clf, scaler, pca = load_models()
        
        # Load the CSV file
        df = pd.read_csv(csv_path)
        #df = df.dropna(subset=['Mudra'])  # Drop rows with NaN values
        df = df[~df['Mudra'].isin(['Vitarka', 'Dharmachakra', 'Bhumisparsa'])]
        left_hand_columns = ['L_PositionX', 'L_PositionY', 'L_PositionZ', 'L_RotationX', 'L_RotationY', 
                     'L_RotationZ', 'L_RotationW', 'L_CurlIndex', 'L_CurlMiddle', 'L_CurlRing', 
                     'L_CurlPinky', 'L_CurlThumb']
        df = df.drop(columns=left_hand_columns, errors='ignore')  # Drop left hand columns
        
        # Initialize counters
        success_count = 0
        fail_count = 0
        
        # Perform multiple random predictions
        for i in range(num_tests):
            # Get a random row
            random_index = np.random.randint(0, len(df))
            sample = df.iloc[random_index]
            
            # Make prediction
            prediction = predict_mudra(sample, autoencoder, clf, scaler, pca)
            
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

def test_with_sample_from_raw_data(raw_data_string):
    """
    Test the prediction function using raw data received from a POST request

    Parameters:
    raw_data_string: str - comma-separated string of feature values

    Returns:
    prediction: str - predicted class
    """
    try:
        # Load models
        autoencoder, clf, scaler, pca = load_models()

        # Convert raw string to numpy array
        values = list(map(float, raw_data_string.strip().split(',')))
        sample = np.array(values)  # ← garante shape (1, N)
        # Convert numpy array to pandas Series
        sample = pd.Series(sample.flatten())
        print("Sample shape:", sample.shape)
        print("Sample values:", sample.values)

        # Make prediction
        prediction = predict_mudra(sample, autoencoder, clf, scaler, pca)

        print("Prediction from raw data:", prediction)
        return prediction
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None


# Example usage
if __name__ == "__main__":
#     # Load a sample from the combined data CSV
    test_with_sample_from_csv(r'Hand_classifier\combined_one_hand_data_with_classification.csv', num_tests=10000)