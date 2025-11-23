import requests
import json

with open('eval/eval_extraction_data.json', 'r') as f:
    eval_extraction_data = json.load(f)

def insert_texts(texts: list[str]) -> None:
    url = "http://localhost:9621/documents/texts"

    payload = {
        "texts": texts
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        print("Texts inserted successfully.")
    else:
        print("Error:", response.status_code, response.text)


texts_to_insert = [document['text'] for document in eval_extraction_data['documents']]
insert_texts(texts_to_insert)