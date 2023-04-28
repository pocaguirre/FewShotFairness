import os

import logging

from typing import Iterable, List, Dict, Any

import backoff

import openai

from tqdm import tqdm

from .apimodel import APIModel

logger = logging.getLogger(__name__ + ".models")
logging.getLogger("openai").setLevel(logging.WARNING)


class ChatGPT(APIModel):
    def __init__(self, model_name: str, temperature: float = 1, max_tokens: int = 5):
        """ChatGPT initializer

        :param model_name: name of model
        :type model_name: str
        :param temperature: temperature of model when generating, defaults to 1
        :type temperature: float, optional
        :param max_tokens: maximum number of tokens generated, defaults to 5
        :type max_tokens: int, optional
        """

        super().__init__(model_name, temperature, max_tokens)

        openai.api_key = os.environ["OPENAI_API_KEY"]

    @backoff.on_exception(
        backoff.expo,
        (
            openai.error.RateLimitError,
            openai.error.APIError,
            openai.error.Timeout,
            openai.error.ServiceUnavailableError,
        ),
    )
    def get_response(self, prompt: str) -> Dict[str, Any]:
        """Send request to ChatGPT API with prompt

        :param prompt: prompt to send to model
        :type prompt: str
        :return: response of API endpoint
        :rtype: Dict[str, Any]
        """
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
        """Clean up response from chatGPT API and return generated string

        :param response: response from chatGPT API
        :type response: Dict[str, Any]
        :return: generated string
        :rtype: str
        """
        text = response["message"]["content"].replace("\n", " ").strip()
        return text

    def generate_from_prompts(self, examples: Iterable[str]) -> List[str]:
        """Send all examples to chatGPT and get its responses

        :param examples: list of prompts
        :type examples: Iterable[str]
        :return: list of cleaned responses
        :rtype: List[str]
        """
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
            except Exception as e:
                print(e)
                responses.append("")
                print(f"Failure of {example}")

        return responses
