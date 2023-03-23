import os

import logging

from typing import Iterable, List, Dict, Any

import openai
from tqdm import tqdm

from .apimodel import APIModel

logger = logging.getLogger(__name__ + ".models")
logging.getLogger("openai").setLevel(logging.WARNING)


class GPT(APIModel):
    """Code modified from
    https://github.com/isabelcachola/generative-prompting/blob/main/genprompt/models.py
    """
    def __init__(self, model_name: str, temperature: float = 1, max_tokens: int = 5):

        super().__init__(model_name, temperature, max_tokens)

        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.batch_size = 32


    def get_response(self, prompt: Iterable[str]) -> Dict[str, Any]:
        """Overloaded get_response to deal with batching

        :param prompt: prompts as batch
        :type prompt: Iterable[str]
        :return: responses from GPT3 API endpoint
        :rtype: Dict[str, Any]
        """        
        response = openai.Completion.create(
            model=self.model_name, prompt=prompt, temperature=self.temperature, max_tokens=self.max_tokens
        )

        return response

    def format_response(self, response: Dict[str, Any]) -> str:
        text = response["text"].replace("\n", " ").strip()
        return text

    def generate_from_prompts(self, examples: Iterable[str]) -> List[str]:
        lines_length = len(examples)
        logger.info(f"Num examples = {lines_length}")
        i = 0

        responses = []

        
        for i in tqdm(range(0, lines_length, self.batch_size), ncols=0):

            # batch prompts together
            prompt_batch = examples[i : min(i + self.batch_size, lines_length)]
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
                        _r = self.get_response(prompt_batch[i][l_prompt - 2000 :])["choices"][0]
                        line = self.format_response(_r)
                        responses.append(line)

        return responses
