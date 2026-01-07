import requests
import json

def teste_envio_dados():
    url = 'https://localhost:5000/predict-mudra'
    dado = "0.1598461,0.8866271,0.2972779,-0.6671056,0.09150723,0.109362,0.7311885,5.960464E-08,0,0,0"

    response = requests.post(url, data=dado)
    print(response.json())


if __name__ == "__main__":
    # Chama a função de teste para enviar dados
    teste_envio_dados()
    # teste_envio_dados()  # Descomente para testar com dados não processados