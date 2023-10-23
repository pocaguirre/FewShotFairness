import csv

import os

import requests

from typing import Iterable, List, Dict, Any

import backoff

import pandas as pd

from tqdm import tqdm

from .apimodel import apimodel


class hf(apimodel):
    def __init__(self, model_name: str, temperature: float = 1, max_tokens: int = 5):
        """HF model initializer

        :param model_name: name of model
        :type model_name: str
        :param temperature: temperature of model when generating, defaults to 1
        :type temperature: float, optional
        :param max_tokens: maximum number of tokens generated, defaults to 5
        :type max_tokens: int, optional
        """

        super().__init__(model_name, temperature, max_tokens)

        self.api_key = os.getenv("HF_ACCESS_TOKEN")
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    @backoff.on_exception(
        backoff.expo,
        (Exception,),
    )
    def get_response(self, prompt: str) -> Dict[str, Any]:
        """Send request to HF API with prompt

        :param prompt: prompt to send to model
        :type prompt: str
        :return: response of API endpoint
        :rtype: Dict[str, Any]
        """
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": self.temperature,
                "max_new_tokens": self.max_tokens,
            },
            "options": {"wait_for_model": True},
        }

        response = requests.post(self.model_name, headers=self.headers, json=payload)

        print(response.json())

        return response.json()[0]

    def format_response(self, response: Dict[str, Any]) -> str:
        """Clean up response from HF API and return generated string

        :param response: response from HF API
        :type response: Dict[str, Any]
        :return: generated string
        :rtype: str
        """
        text = response["generated_text"].replace("\n", " ").strip()
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
        """Send all examples to HF model API and get its responses

        :param examples: list of prompts
        :type examples: Iterable[str]
        :return: list of cleaned responses
        :rtype: List[str]
        """
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
            # loop through examples provided
            for i in range(checkpoint_start, len(prompts)):
                prompt = prompts[i]
                label = labels[i]
                demographic = demographics[i]

                # try to get response
                # catch exceptions that happen
                response = self.get_response(prompt)
                formatted_response = self.format_response(response)

                csvwriter.writerow([prompt, response, label, demographic])

                responses.append(formatted_response)

        return responses
