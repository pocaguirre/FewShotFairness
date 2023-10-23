import csv

import os

import logging

from typing import Iterable, List, Dict, Any

import backoff

import openai

import pandas as pd

from tqdm import tqdm

from .apimodel import apimodel

logger = logging.getLogger(__name__ + ".models")
logging.getLogger("openai").setLevel(logging.WARNING)


class chatgpt(apimodel):
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

    def generate_from_prompts(
        self,
        prompts: Iterable[str],
        output_folder: str,
        model_name: str,
        dataset: str,
        demonstration: str,
        test_df: pd.DataFrame,
        checkpoint_start: int,
    ) -> List[str]:
        """Send all prompts to chatGPT and get its responses

        :param prompts: list of prompts
        :type prompts: Iterable[str]
        :return: list of cleaned responses
        :rtype: List[str]
        """
        lines_length = len(prompts)
        logger.info(f"Num prompts = {lines_length}")

        responses = []

        labels = test_df["labels"].tolist()
        demographics = test_df["demographics"].tolist()

        if os.path.exists(
            os.path.join(output_folder, f"{model_name}_{dataset}_{demonstration}.csv")
        ):
            mode = "a"
        else:
            mode = "w"
        
        with open(
            os.path.join(output_folder, f"{model_name}_{dataset}_{demonstration}.csv"),
            mode,
        ) as csvfile:
            csvwriter = csv.writer(csvfile)
            if mode == "w":
                csvwriter.writerow(["prompt", "response", "label", "demographic"])

            # loop through examples
            for i in tqdm(range(checkpoint_start ,len(prompts))):
                # try to get response
                # catch any errors that happen
                prompt = prompt[i]
                label = labels[i]
                demographic = demographics[i]

                try:
                    response = self.get_response(prompt)
                    formatted_response = self.format_response(response["choices"][0])
                    responses.append(formatted_response)
                    
                except Exception as e:
                    print(e)
                    formatted_response = ""
                    responses.append(formatted_response)
                    print(f"Failure of {prompt}")
                
                csvwriter.writerow([prompt, formatted_response, label, demographic])

        return responses
