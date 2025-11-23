import requests

from langchain_core.tools import tool


def call_sysprompt_api(system_prompt: str, query: str) -> dict:
    url = "http://localhost:9621/query/sysprompt"

    header = {
        "Content-Type": "application/json",
        "accept": "application/json"
    }

    payload = {
        "request": {
            "query": query,
            "mode": "naive",
            "top_k": 1,
            "chunk_top_k": 1,
            "max_total_tokens": 9000,
            "history_turns": 0,
            "enable_rerank": False
        },
        "sys_prompt": system_prompt
    }

    response = requests.post(url, json=payload, headers=header)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API call failed with status code {response.status_code}: {response.text}")