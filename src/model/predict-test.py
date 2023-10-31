import requests

transaction = {
            'score_1': 4,
            'score_2': 0.7685,
            'score_3': 94436.24,
            'score_4': 20.0,
            'score_5': 0.444828,
            'score_6': 1.0,
            'pais': 'BR',
            'score_7': 5,
            'produto': 'Máquininha Corta Barba',
            'categoria_produto': 'cat_8d714cd',
            'score_8': 0.883598,
            'score_9': 240,
            'score_10': 102.0,
            'entrega_doc_1': 1,
            'entrega_doc_2': 'Y',
            'entrega_doc_3': 'N',
            'data_compra': '2020-03-27 11:51:16',
            'valor_compra': 5.64,
            'score_fraude_modelo': 66
                }

url = "http://localhost:9696/predict"

request = requests.post(url, json=transaction)

print(request.json())
