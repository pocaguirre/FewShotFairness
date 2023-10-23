from typing import Dict, Any, List, Iterable

import pandas as pd


class apimodel:
    def __init__(
        self, model_name: str, temperature: float = 1, max_tokens: int = 5
    ) -> None:
        """Base model for LLMs

        :param model_name: name of model
        :type model_name: str
        :param temperature: temperature of model when generating, defaults to 1
        :type temperature: float, optional
        :param max_tokens: maximum number of tokens generated, defaults to 5
        :type max_tokens: int, optional
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def get_response(self, prompt: str) -> Dict[str, Any]:
        """Send request to API with prompt

        :param prompt: prompt to send to model
        :type prompt: str
        :return: response of API endpoint
        :rtype: Dict[str, Any]
        """
        pass

    def format_response(self, response: Dict[str, Any]) -> str:
        """Clean up response from API and return generated string

        :param response: response from LLM API
        :type response: Dict[str, Any]
        :return: generated string
        :rtype: str
        """
        pass

    def generate_from_prompts(
        self,
        prompts: Iterable[str],
        output_folder: str,
        model_name: str,
        dataset: str,
        demonstration: str,
        test_df: pd.DataFrame,
        checkpoint: int,
    ) -> List[str]:
        """Send all prompts to model and get and clean their resposnes

        :param prompts: list of prompts
        :type prompts: Iterable[str]
        :return: list of cleaned responses
        :rtype: List[str]
        """
        pass
