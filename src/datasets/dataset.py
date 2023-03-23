from typing import Tuple, List


class Dataset:
    """Base Class for datasets
    """    
    def __init__(self, path: str):
        self.path = path

    def build_prompt(self, text: str, label: str) -> str:
        """Create prompt from input text and label

        :param text: input text for dataset
        :type text: str
        :param label: classification label for input text
        :type label: str
        :return: prompt containing text and label
        :rtype: str
        """        
        pass

    def create_prompts(self) -> Tuple[List[str], List[str], List[str], List[List[str]]]:
        """Creates prompt from train and test datasets

        :return: Tuple of training prompts, testing prompts, test labels, and demographics for test set 
        :rtype: Tuple[List[str], List[str], List[str], List[List[str]]]
        """        
        pass
