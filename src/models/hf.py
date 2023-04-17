import os

import requests

from typing import Iterable, List, Dict, Any

import backoff

from tqdm import tqdm

from .apimodel import APIModel


class HF(APIModel):
    def __init__(self, model_name: str, temperature: float = 1, max_tokens: int = 5):

        super().__init__(model_name, temperature, max_tokens)

        self.api_key = os.getenv("HF_ACCESS_TOKEN")
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
    
    @backoff.on_exception(
        backoff.expo,
        (
            Exception,
        ),
    )
    def get_response(self, prompt: str) -> Dict[str, Any]:
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": self.temperature,
                "max_new_tokens": self.max_tokens,
            },
            "options": {"wait_for_model": True}
        }

        response = requests.post(self.model_name, headers=self.headers, json=payload)

        print(response)

        return response.json()[0]

    def format_response(self, response: Dict[str, Any]) -> str:
        text = response["generated_text"].replace("\n", " ").strip()
        return text

    def generate_from_prompts(self, examples: Iterable[str]) -> List[str]:
        responses = []

        # loop through examples provided
        for example in tqdm(examples):

            # try to get response
            # catch exceptions that happen
            response = self.get_response(example)
            formatted_response = self.format_response(response)
            responses.append(formatted_response)

        return responses
