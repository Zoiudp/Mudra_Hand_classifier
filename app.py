from flask import Flask, request
from Hand_classifier import test_with_sample_from_raw_data
import os

app = Flask(__name__)

@app.route('/receber-dado', methods=['POST'])
def receber_dado():
    # Captura o corpo bruto da requisição
    dado_bruto = request.data.decode('utf-8')  # transforma bytes em string
    print("Dado recebido:", dado_bruto)
    
    # Transforma a string em lista de floats
    lista_de_floats = list(map(float, dado_bruto.split(',')))
    print("Lista de floats:", lista_de_floats)
    
    return {'status': 'ok', 'valores_recebidos': lista_de_floats}, 200

@app.route('/predict-mudra', methods=['POST'])
def predict_mudra_from_post():
    raw_data = request.data.decode('utf-8')
    prediction = test_with_sample_from_raw_data(raw_data)
    return {'prediction': prediction}, 200


if __name__ == '__main__':
    # Define o caminho do certificado e da chave privada
    app.run(ssl_context=('Hand_classifier\certificado.crt', 'Hand_classifier\chave_privada.key'))