import random

from typing import List, Iterable, Tuple

from tqdm import tqdm

class RandomSampler:
    
    def __init__(self, seed = 42, shots = 16):
        random.seed(seed)
        self.shots = shots
    
    def create_demonstrations(self, train_set: Iterable[str], test_set: Iterable[str]) -> List[str]:
        
        demonstrations = []

        for item in tqdm(test_set):
            train_dems = random.sample(train_set, self.shots)

            demonstrations.append(train_dems + '\n' + item)
        
        return demonstrations
