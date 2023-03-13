from typing import Tuple, List

class Dataset:

    def __init__(self, path: str, prompt: str):

        self.path = path

        self.prompt = prompt

    def create_prompts(self) -> Tuple[List[str], List[str], List[str]]:
        pass