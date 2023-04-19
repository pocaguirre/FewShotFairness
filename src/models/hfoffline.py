import os

import torch

from typing import Iterable, List, Dict, Any

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from tqdm import tqdm

from .apimodel import APIModel

class HFOffline(APIModel):
    def __init__(self, model_name: str, temperature: float = 1, max_tokens: int = 5):
        super().__init__(model_name, temperature, max_tokens)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

        self.batch_size = 20


    def get_response(self, prompts: Iterable[str]) -> Dict[str, Any]:

        tokenized_input = self.tokenizer(prompts).to(self.device)

        outputs = self.model.generate(tokenized_input, temperature = self.temperature, max_new_tokens = self.max_tokens)
        
        return self.tokenizer.batch_decode(outputs.cpu(), skip_special_tokens=True)

    def format_response(self, response: str) -> str:
        text = response.replace("\n", " ").strip()
        return text

    def generate_from_prompts(self, examples: Iterable[str]) -> List[str]:
        responses = []

        for i in tqdm(range(0, len(examples), self.batch_size), ncols=0):

            prompt_batch = examples[i : min(i + self.batch_size, len(examples))]

            response = self.get_response(prompt_batch)

            response = [self.format_response(x) for x in response]

            responses.extend(response)

        return responses

