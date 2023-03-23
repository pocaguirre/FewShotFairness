import random

from typing import List, Iterable, Tuple

from tqdm import tqdm

class RandomSampler:
    def __init__(self, shots=16) -> None:

        self.shots = shots

    def create_demonstrations(
        self, train_set: Iterable[str], test_set: Iterable[str]
    ) -> List[str]:
        """Create random k-shot from train set and test set

        :param train_set: list of train prompts
        :type train_set: Iterable[str]
        :param test_set: list of test prompts
        :type test_set: Iterable[str]
        :return: k-shot demonstrations for each test set item 
        :rtype: List[str]
        """        
        demonstrations = []

        for item in tqdm(test_set):
            train_dems = random.sample(train_set, self.shots)

            demonstrations.append("\n\n".join(train_dems) + "\n\n" + item)

        return demonstrations