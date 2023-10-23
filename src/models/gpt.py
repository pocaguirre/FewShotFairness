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


class gpt(apimodel):
    """Code modified from
    https://github.com/isabelcachola/generative-prompting/blob/main/genprompt/models.py
    """

    def __init__(self, model_name: str, temperature: float = 1, max_tokens: int = 5):
        """ "GPT model initializer

        :param model_name: name of model
        :type model_name: str
        :param temperature: temperature of model when generating, defaults to 1
        :type temperature: float, optional
        :param max_tokens: maximum number of tokens generated, defaults to 5
        :type max_tokens: int, optional
        """
        super().__init__(model_name, temperature, max_tokens)

        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.batch_size = 20

    @backoff.on_exception(
        backoff.expo,
        (
            openai.error.RateLimitError,
            openai.error.APIError,
            openai.error.Timeout,
            openai.error.ServiceUnavailableError,
        ),
    )
    def get_response(self, prompt: Iterable[str]) -> Dict[str, Any]:
        """Overloaded get_response to deal with batching

        :param prompt: prompts as batch
        :type prompt: Iterable[str]
        :return: responses from GPT3 API endpoint
        :rtype: Dict[str, Any]
        """
        response = openai.Completion.create(
            model=self.model_name,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return response

    def format_response(self, response: Dict[str, Any]) -> str:
        """Clean up response from GPT API and return generated string

        :param response: response from GPT API
        :type response: Dict[str, Any]
        :return: generated string
        :rtype: str
        """
        text = response["text"].replace("\n", " ").strip()
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
        """Send all prompts to GPT model API and get its responses

        :param prompts: list of prompts
        :type prompts: Iterable[str]
        :return: list of cleaned responses
        :rtype: List[str]
        """
        lines_length = len(prompts)
        logger.info(f"Num examples = {lines_length}")
        i = 0

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
            for i in tqdm(range(checkpoint_start* self.batch_size, lines_length, self.batch_size), ncols=0):
                # batch prompts together
                prompt_batch = prompts[i : min(i + self.batch_size, len(prompts))]
                label_batch = labels[i : min(i + self.batch_size, len(labels))]
                demographics_batch = demographics[
                    i : min(i + self.batch_size, len(demographics))
                ]

                try:
                    # try to get respones
                    response = self.get_response(prompt_batch)

                    response_batch = [""] * len(prompt_batch)

                    # order the responses as they are async
                    for choice in response.choices:
                        response_batch[choice.index] = self.format_response(choice.text)

                    responses.extend(response_batch)

                # catch any connection exceptions
                except:
                    # try each prompt individually
                    for i in range(len(prompt_batch)):
                        try:
                            _r = self.get_response(prompt_batch[i])["choices"][0]
                            line = self.format_response(_r)
                            responses.append(line)
                        except:
                            # if there is an exception make blank
                            l_prompt = len(prompt_batch[i])
                            _r = self.get_response(prompt_batch[i][l_prompt - 2000 :])[
                                "choices"
                            ][0]
                            line = self.format_response(_r)
                            responses.append(line)
                
                out = list(
                        zip(
                            prompt_batch,
                            response_batch,
                            label_batch,
                            demographics_batch,
                        )
                    )

                csvwriter.writerows(out)

        return responses
