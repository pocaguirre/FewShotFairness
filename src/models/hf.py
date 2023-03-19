import json

import os

import requests

from typing import Iterable, List

from tqdm import tqdm


class HF:
    def __init__(self, model_url: str):
        self.api_key = os.getenv("HF_ACCESS_TOKEN")
        self.model_url = model_url
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def get_response(self, prompt: str, temperature=0):
        payload = json.dumps(
            {"inputs": prompt, "parameters": {"temperature": temperature}}
        )

        response = requests.post(self.model_url, headers=self.headers, json=payload)

        return json.loads(response.content.decode("utf-8"))

    def format_response(self, response):
        text = response["text"].replace("\n", " ").strip()
        return text

    def generate_from_prompts(self, examples: Iterable[str]) -> List[str]:
        responses = []

        for example in tqdm(examples):
            try:
                response = self.get_response(example)["generated_text"]
                formatted_response = self.format_response(response)
                responses.append(formatted_response + "\n")
            except:
                continue

        return responses
