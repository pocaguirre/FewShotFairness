import os

import torch

from typing import Iterable, List, Dict, Any

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)

from tqdm import tqdm

from .apimodel import apimodel


class hfoffline(apimodel):
    def __init__(self, model_name: str, temperature: float = 1, max_tokens: int = 5):
        """HF offline model initializer

        :param model_name: name of model
        :type model_name: str
        :param temperature: temperature of model when generating, defaults to 1
        :type temperature: float, optional
        :param max_tokens: maximum number of tokens generated, defaults to 5
        :type max_tokens: int, optional
        """
        super().__init__(model_name, temperature, max_tokens)

        self.model = None

        if self.model_name in [
            "chavinlo/alpaca-13b",
            "huggyllama/llama-13b",
            "huggyllama/llama-65b",
            "chavinlo/alpaca-native",
        ]:
            n_gpus = torch.cuda.device_count()

            if n_gpus == 1:

                self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(0)
            
            else:

                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map = "balanced_low_0")


        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(0)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, model_max_length=2048
        )

        if self.model_name in ["huggyllama/llama-13b", "huggyllama/llama-65b"]:
            self.tokenizer.pad_token = '<unk>'
        
        elif self.model_name in ["chavinlo/alpaca-13b", "chavinlo/alpaca-native"]:
            self.tokenizer.pad_token = '[PAD]'

        self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.eval()
        
        self.batch_size = 1

    def get_response(self, prompts: Iterable[str]) -> Dict[str, Any]:
        """ "Get response from HF model with prompt batch

        :param prompt: prompt to send to model
        :type prompt: Iterable[str]
        :return: response of API endpoint
        :rtype: Dict[str, Any]
        """
        tokenized_input = self.tokenizer(
            prompts, return_tensors="pt", padding="max_length"
        )

        if self.model_name in [
            "chavinlo/alpaca-13b",
            "huggyllama/llama-13b",
            "huggyllama/llama-65b",
            "chavinlo/alpaca-native",
        ]:
            del tokenized_input["token_type_ids"]

        tokenized_input.to(0)

        outputs = self.model.generate(
            **tokenized_input,
            temperature=self.temperature,
            max_new_tokens=self.max_tokens
        )

        output = self.tokenizer.batch_decode(outputs.cpu(), skip_special_tokens=True)

        return output

    def format_response(self, response: str) -> str:
        """Clean up response from Offline HF model and return generated string

        :param response: response from Offline HF model
        :type response: Dict[str, Any]
        :return: generated string
        :rtype: str
        """
        text = response.replace("\n", " ").strip()
        return text

    def generate_from_prompts(self, examples: Iterable[str]) -> List[str]:
        """Send all examples to offline HF model and get its responses

        :param examples: list of prompts
        :type examples: Iterable[str]
        :return: list of cleaned responses
        :rtype: List[str]
        """
        responses = []

        with torch.inference_mode():
            for i in tqdm(range(0, len(examples), self.batch_size), ncols=0):
                prompt_batch = examples[i : min(i + self.batch_size, len(examples))]

                response = self.get_response(prompt_batch)

                response = [self.format_response(x) for x in response]

                responses.extend(response)

        del self.model
        torch.cuda.empty_cache()

        return responses
