import torch
import joblib
import pandas as pd
import numpy as np
from Autoencoder import Autoencoder # Assuming Autoencoder.py contains the Autoencoder class
import os
# Carregar o modelo treinado, scaler e label_encoder
from tensorflow.keras.models import load_model
import pickle

def test_with_sample_from_raw_data_lstm(raw_data_string):
    """
    Test the prediction function using raw data received from a POST request

    Parameters:
    raw_data_string: str - comma-separated string of feature values

    Returns:
    prediction: str - predicted class
    """
    try:
        

        model_path = "vr_gesture_model.h5"
        scaler_path = "vr_gesture_scaler.pkl"
        encoder_path = "vr_gesture_encoder.pkl"

        # Verificar se os arquivos existem
        if not all(os.path.exists(path) for path in [model_path, scaler_path, encoder_path]):
            print("Arquivos do modelo não encontrados.")
            return None

        # Carregar modelo e transformadores
        model = load_model(model_path)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)

        # Configurações - devem corresponder às mesmas usadas no treinamento
        WINDOW_SIZE = 20
        FEATURES = ['R_PositionX', 'R_PositionY', 'R_PositionZ', 
                   'R_RotationX', 'R_RotationY', 'R_RotationZ', 'R_RotationW',
                   'R_CurlIndex', 'R_CurlMiddle', 'R_CurlRing', 'R_CurlPinky', 'R_CurlThumb']
        
        # Converter raw string para arrays numpy
        rows = raw_data_string.strip().split('\n')
        
        # Verificar se há dados suficientes para uma janela
        if len(rows) < WINDOW_SIZE:
            print(f"Dados insuficientes. Necessário pelo menos {WINDOW_SIZE} registros, recebidos {len(rows)}.")
            return None
        
        # Pegar apenas os dados mais recentes (último WINDOW_SIZE)
        rows = rows[-WINDOW_SIZE:]
        
        # Processar cada linha
        data_list = []
        for row in rows:
            values = list(map(float, row.strip().split(',')))
            # Assumindo que os valores estão na ordem esperada (sem incluir o alvo)
            if len(values) >= len(FEATURES):
                # Se houver valores extras, pegar apenas os features relevantes
                feature_values = values[:len(FEATURES)]
                data_list.append(feature_values)
            else:
                print(f"Linha com dados insuficientes: {row}")
                return None
        
        # Converter para numpy array
        window_data = np.array(data_list)
        print("Window shape:", window_data.shape)
        
        # Normalizar os dados
        window_data_scaled = scaler.transform(window_data)
        
        # Reshape para o formato esperado pelo modelo LSTM (batch_size, timesteps, features)
        input_data = window_data_scaled.reshape(1, WINDOW_SIZE, len(FEATURES))
        
        # Fazer a previsão
        prediction_probs = model.predict(input_data, verbose=0)[0]
        predicted_class_idx = np.argmax(prediction_probs)
        confidence = prediction_probs[predicted_class_idx] * 100
        
        # Converter índice para o nome da classe
        predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
        
        print(f"Previsão: {predicted_class} (Confiança: {confidence:.2f}%)")
        return predicted_class
        
    except Exception as e:
        import traceback
        print(f"Erro durante a previsão: {e}")
        print(traceback.format_exc())
        return None

def load_models():
    # Load the trained models and scaler
    try:
        # Defina o caminho para o diretório onde o arquivo está localizado
        diretorio = 'Hand_classifier'

        # Construa o caminho completo para o arquivo .pth
        caminho_autoencoder = os.path.join(diretorio, 'autoencoder.pth')
        caminho_random_forest = os.path.join(diretorio, 'mudra_random_forest.joblib')
        caminho_scaler = os.path.join(diretorio, 'mudra_scaler.joblib')
        caminho_tsne = os.path.join(diretorio, 'mudra_tsne.joblib')

        # Load the autoencoder model
        autoencoder = Autoencoder(input_dim=24, latent_dim=10)
        autoencoder.load_state_dict(torch.load(caminho_autoencoder))
        autoencoder.eval()  # Set to evaluation mode
        
        # Load the Random Forest classifier
        clf = joblib.load(caminho_random_forest)
        
        # Load the scaler
        scaler = joblib.load(caminho_scaler)
        
        return autoencoder, clf, scaler
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None

def predict_mudra(data, autoencoder=None, clf=None, scaler=None, tsne=None):
    """
    Predict the Mudra class for a single data point
    
    Parameters:
    data: pandas.Series or numpy.ndarray - the input data point
    autoencoder, clf, scaler: pre-loaded models (optional)
    
    Returns:
    predicted_class: str - the predicted Mudra class
    """
    # Load models if not provided
    if autoencoder is None or clf is None or scaler is None or tsne is None:
        autoencoder, clf, scaler, tsne = load_models()
    
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
    
    latent_repr_tsne = tsne.transform(latent_repr)
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
        autoencoder, clf, scaler, tsne = load_models()

        # Convert raw string to numpy array
        values = list(map(float, raw_data_string.strip().split(',')))
        sample = np.array(values)  # ← garante shape (1, N)
        # Convert numpy array to pandas Series
        sample = pd.Series(sample.flatten())
        print("Sample shape:", sample.shape)
        print("Sample values:", sample.values)

        # Make prediction
        prediction = predict_mudra(sample, autoencoder, clf, scaler, tsne)

        print("Prediction from raw data:", prediction)
        return prediction
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None


# Example usage
# if __name__ == "__main__":
#     # Load a sample from the combined data CSV
#     test_with_sample_from_csv('combined_data_with_classification.csv')