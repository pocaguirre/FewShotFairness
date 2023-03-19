from typing import Tuple, List


class Dataset:
    def __init__(self, path: str):
        self.path = path

    def build_prompt(self, text: str, label: str) -> str:
        pass

    def create_prompts(self) -> Tuple[List[str], List[str], List[str], List[List[str]]]:
        pass
