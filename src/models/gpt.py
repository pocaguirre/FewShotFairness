import os

import logging

from typing import Iterable, List 

import openai
from tqdm import tqdm 

logger = logging.getLogger(__name__ +'.models')
logging.getLogger("openai").setLevel(logging.WARNING)


class GPT:
    def __init__(self, model_name: str, batch_size: int = 32):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.batch_size = batch_size    
        
    def get_response(self, prompt: Iterable[str], temperature=0):
        response = openai.Completion.create(model=self.model_name, 
                                            prompt=prompt, 
                                            temperature=temperature,
                                            max_tokens=1
                                            )
        return response

    def format_response(self, response):
        text = response['text'].replace('\n', ' ').strip()
        return text

    def generate_from_prompts(self, examples: Iterable[str]) -> List[str]:
        lines_length = len(examples)
        logger.info(f'Num examples = {lines_length}')
        i = 0

        responses =  []

        for i in tqdm(range(0, lines_length, self.batch_size), ncols=0):
            prompt_batch = examples[i:min(i+self.batch_size, lines_length)]
            try:
                response = self.get_response(prompt_batch)
                for line in response['choices']:
                    line = self.format_response(line)
                    responses.append(line + '\n')
            except:
                for i in range(len(prompt_batch)):
                    try:
                        _r = self.get_response(prompt_batch[i])['choices'][0]
                        line = self.format_response(_r)
                        responses.append(line + '\n')
                    except:
                        l_prompt = len(prompt_batch[i])
                        _r = self.get_response(prompt_batch[i][l_prompt-2000:])['choices'][0]
                        line = self.format_response(_r)
                        responses.append(line + '\n')

        return responses