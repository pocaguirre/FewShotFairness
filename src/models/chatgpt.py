import os

import logging

from typing import Iterable, List, Dict, Any

import openai
from tqdm import tqdm

from .apimodel import APIModel

logger = logging.getLogger(__name__ + ".models")
logging.getLogger("openai").setLevel(logging.WARNING)


class ChatGPT(APIModel):
    def __init__(self, model_name: str, temperature: float = 1, max_tokens: int = 5):

        super.__init__(model_name, temperature, max_tokens)

        openai.api_key = os.environ["OPENAI_API_KEY"]

    def get_response(self, prompt: str) -> Dict[str, Any]:
        messages = [
            {"role": "user", "content": prompt},
        ]

        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response

    def format_response(self, response: Dict[str, Any]) -> str:
        text = response["message"]["content"].replace("\n", " ").strip()
        return text

    def generate_from_prompts(self, examples: Iterable[str]) -> List[str]:
        lines_length = len(examples)
        logger.info(f"Num examples = {lines_length}")

        responses = []

        # loop through examples 
        for example in tqdm(examples):

            # try to get response
            # catch any errors that happen
            try:
                response = self.get_response(example)
                for line in response["choices"]:
                    line = self.format_response(line)
                    responses.append(line)
            except:
                continue

        return responses
