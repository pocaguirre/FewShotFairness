import csv

import os

from typing import Iterable, List, Dict, Any

import pandas as pd

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)

from tqdm import tqdm

from .apimodel import apimodel

import subprocess as sp
import os


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = (
        sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    )
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


class hfoffline(apimodel):
    def __init__(self, model_name: str, temperature: float, max_tokens: int = 5):
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
            "meta-llama/Llama-2-13b-hf",
            "meta-llama/Llama-2-70b-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-70b-chat-hf",
        ]:
            n_gpus = torch.cuda.device_count()

            if n_gpus == 1:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(0)

            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, device_map="auto", torch_dtype=torch.float16
                )

        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(0)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.model_name in ["huggyllama/llama-13b", "huggyllama/llama-65b"]:
            self.tokenizer.pad_token = "<unk>"
            self.tokenizer.pad_token_id = 0

        elif self.model_name in ["chavinlo/alpaca-13b", "chavinlo/alpaca-native"]:
            self.tokenizer.pad_token = "[PAD]"
            self.tokenizer.pad_token_id = 0

        self.tokenizer.pad_token = "[PAD]"
        self.tokenizer.padding_side = "left"

        self.model.eval()

        self.batch_size = 1

    def get_response(self, prompts: Iterable[str]) -> Dict[str, Any]:
        """ "Get response from HF model with prompt batch

        :param prompt: prompt to send to model
        :type prompt: Iterable[str]
        :return: response of API endpoint
        :rtype: Dict[str, Any]
        """
        tokenized_input = self.tokenizer(prompts, return_tensors="pt", padding=True)

        if self.model_name in [
            "chavinlo/alpaca-13b",
            "huggyllama/llama-13b",
            "huggyllama/llama-65b",
            "chavinlo/alpaca-native",
        ]:
            del tokenized_input["token_type_ids"]

        outputs = self.model.generate(
            tokenized_input.input_ids.to(0),
            temperature=self.temperature,
            max_new_tokens=self.max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        del tokenized_input

        output = self.tokenizer.batch_decode(outputs.cpu(), skip_special_tokens=True)

        del outputs

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
        """Send all examples to offline HF model and get its responses

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
            with torch.inference_mode():
                for i in tqdm(range(checkpoint_start * self.batch_size, len(prompts), self.batch_size), ncols=0):
                    prompt_batch = prompts[i : min(i + self.batch_size, len(prompts))]
                    label_batch = labels[i : min(i + self.batch_size, len(labels))]
                    demographics_batch = demographics[
                        i : min(i + self.batch_size, len(demographics))
                    ]

                    response = self.get_response(prompt_batch)

                    response_batch = [self.format_response(x) for x in response]

                    out = list(
                        zip(
                            prompt_batch,
                            response_batch,
                            label_batch,
                            demographics_batch,
                        )
                    )

                    csvwriter.writerows(out)

                    responses.extend(response)

        del self.model
        torch.cuda.empty_cache()

        return responses
