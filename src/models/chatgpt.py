import os

import logging

from typing import Iterable, List

import openai
from tqdm import tqdm

logger = logging.getLogger(__name__ + ".models")
logging.getLogger("openai").setLevel(logging.WARNING)


class ChatGPT:
    def __init__(self, model_name: str):
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.model_name = model_name

    def get_response(self, prompt: str, temperature=1):
        messages = [
            {"role": "user", "content": prompt},
        ]

        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=5,
        )
        return response

    def format_response(self, response):
        text = response["message"]["content"].replace("\n", " ").strip()
        return text

    def generate_from_prompts(self, examples: Iterable[str]) -> List[str]:
        lines_length = len(examples)
        logger.info(f"Num examples = {lines_length}")

        responses = []

        for example in tqdm(examples):
            try:
                response = self.get_response(example)
                for line in response["choices"]:
                    line = self.format_response(line)
                    responses.append(line + "\n")
            except:
                continue

        return responses
