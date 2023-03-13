import argparse

import os

import tomllib

from typing import List, Dict, Any

import random

import numpy as np

from .datasets.biasinbios import BiasInBios
from .datasets.twitteraae import TwitterAAE
from .datasets.sbic import SBIC

from .models.gpt import GPT
from .models.hf import HF

from .demonstrations.random import RandomSampler

def set_randomness(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def build_demonstration(demonstration_name: str, demonstration_params: Dict[str, Any], train: List[str], test: List[str]) -> List[str]:
    shots = None
    
    if demonstration_name == "zeroshot":
        shots = 0
    elif demonstration_name == "random":
        shots = demonstration_params['shots']
    else:
        raise ValueError(f"{demonstration_name} does not exist!")

    sampler = RandomSampler(shots = shots)

    return sampler.create_demonstrations(train, test)


def build_model(model_name: str):
    models = {
        "gpt3" : GPT("text-curie-001"),
        "chatgpt": GPT("gpt-3.5-turbo"),
        "flan" : HF("https://api-inference.huggingface.co/models/google/flan-t5-large"),
        "ul2" : HF("https://api-inference.huggingface.co/models/google/ul2")
    }

    model = None

    try: 
        model = models[model_name]
    except KeyError:
        raise ValueError(f"{model_name} does not exist!")

    return model

def build_dataset(dataset_name: str, path: str):
    datasets = {
        "sbic" : SBIC,
        "aae": TwitterAAE,
        "bias": BiasInBios
    }

    prompts = {
        "aae" : "{text} \n {label}",
        "bias": "{text} \n The occupation of this person is {label}",
        "sbic": "Post:{text}\nQuestion: Does this post cotain offensive language?\nAnswer: {label}",
        "hatexplain" : "{text} \n {label}"
    }

    try:
        dataset = datasets[dataset_name](path, prompts[dataset_name])
    except KeyError:
        raise ValueError(f"{dataset_name} does not exist!")

    return dataset.create_prompts()


def run_dataset(train: List[str], test: List[str], test_labels: List[str], models_and_demonstrations: Dict[str, List[str]]):

    demonstrations = models_and_demonstrations['demonstrations']
    models = models_and_demonstrations['models']

    for model_name in models:
        for demonstration_name in demonstrations:

            model = build_model(model_name)

            demonstration_params = demonstrations[demonstration_name]
            
            input_prompts = build_demonstration(demonstration_name, demonstration_params, train, test)

            #responses = model.generate_from_prompts(input_prompts)


def main(args):

    configpath = args.config

    with open(configpath, "rb") as f:
        data = tomllib.load(f)
    
    randomseed = data['general']['seed']

    datasets = data['datasets']

    set_randomness(randomseed)

    for dataset in datasets:

        train, test, test_labels = build_dataset(dataset, datasets[dataset]['path'])

        run_dataset(train, test, test_labels, datasets[dataset])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
    )

    args = parser.parse_args()

    main(args)