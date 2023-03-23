import argparse

import csv

import logging

import os

import tomllib

from typing import List, Dict, Any

import random

import numpy as np

from .datasets.biasinbios import BiasInBios
from .datasets.twitteraae import TwitterAAE
from .datasets.hatexplain import HatExplain
from .datasets.dataset import Dataset

from .models.gpt import GPT
from .models.chatgpt import ChatGPT
from .models.hf import HF
from .models.apimodel import APIModel

from .demonstrations.random import RandomSampler

from .utils import metrics

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def set_randomness(seed: int):
    """Set the randomness of the entire script

    :param seed: random seed
    :type seed: int
    """    
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    
def build_demonstration(
    demonstration_name: str,
    demonstration_params: Dict[str, Any],
    train: List[str],
    test: List[str],
) -> List[str]:
    """Build demonstrations based on parameters

    :param demonstration_name: name of demonstration (zeroshot, random)
    :type demonstration_name: str
    :param demonstration_params: parameters for the demonstration
    :type demonstration_params: Dict[str, Any]
    :param train: list of training prompts
    :type train: List[str]
    :param test: list of test prompts
    :type test: List[str]
    :raises ValueError: demonstration does not exist
    :return: the list of formed demonstrations
    :rtype: List[str]
    """    

    shots = None

    if demonstration_name == "zeroshot":
        shots = 0
    elif demonstration_name == "random":
        shots = demonstration_params["shots"]
    else:
        raise ValueError(f"{demonstration_name} does not exist!")

    sampler = RandomSampler(shots=shots)

    return sampler.create_demonstrations(train, test)


def build_model(model_name: str, model_params: Dict[str, Any]) -> APIModel:
    """Builds model class from model name and params provided

    :param model_name: name of model being used
    :type model_name: str
    :param model_params: list of parameters provided for each model
    :type model_params: Dict[str, Any]
    :raises ValueError: model does not exist
    :return: fully formed model
    :rtype: APIModel
    """    
    models = {
        "gpt3": GPT("text-curie-001", **model_params),
        "chatgpt": ChatGPT("gpt-3.5-turbo", **model_params),
        "flan": HF("https://api-inference.huggingface.co/models/google/flan-t5-large", **model_params),
        "ul2": HF("https://api-inference.huggingface.co/models/google/ul2", **model_params),
    }

    model = None

    try:
        model = models[model_name]
    except KeyError:
        raise ValueError(f"{model_name} does not exist!")

    return model


def build_dataset(dataset_name: str, path: str) -> Dataset:
    """Build dataset based on name and path to it

    :param dataset_name: name of dataset
    :type dataset_name: str
    :param path: path of dataset
    :type path: str
    :raises ValueError: dataset name does not exist
    :return: Created dataset
    :rtype: Dataset
    """    
    datasets = {
        "hatexplain": HatExplain,
        "aae": TwitterAAE,
        "bias": BiasInBios,
    }

    try:
        dataset = datasets[dataset_name](path)
    except KeyError:
        raise ValueError(f"{dataset_name} does not exist!")

    return dataset.create_prompts()


def run_dataset(
    prompts: List[str],
    test_labels: List[str],
    test_demographics: List[str],
    dataset: str,
    demonstration: str, 
    models: List[str],
):

    for model_name in models:

        logging.info(f"Starting to create {model_name} model for {dataset} with {demonstration}")

        model = build_model(model_name, models[model_name])

        logging.info(f"Created {model_name} model for {dataset} with {demonstration}")

        logging.info(f"Running {model_name} on {dataset} with {demonstration}")

        responses = model.generate_from_prompts(prompts)

        if not os.path.exists("./output/responses"):
            os.makedirs("./output/responses", exist_ok=True)

        with open(os.path.join("./output/responses", f"{model_name}_{dataset}_{demonstration}"), 'w') as csvfile: 
            csvwriter = csv.writer(csvfile) 
        
            csvwriter.writerow(['response', 'label', 'demographic']) 

            for response, label, demographic in zip(responses, test_labels, test_demographics):
                csvwriter.writerow([response, label, demographic])

        logging.info(f"Completed running {model_name} on {dataset} with {demonstration}")

        logging.info(f"Calculating metrics for {model_name} on {dataset} with {demonstration}")

        results = []

        if dataset == "hatexplain":
            results.append(metrics(responses, test_labels, "hatexplain-race", test_demographics))
            results.append(metrics(responses, test_labels, "hatexplain-gender", test_demographics))
        else:
            results.append(metrics(responses, test_labels, dataset, test_demographics))

        for result in results:

            gaps = result['max_gaps']

            group_results = result['score']

            logging.info(f"For {model_name} on {dataset}")
            logging.info(f"F1 Overall: {result['total_score']}")
            for result in group_results:
                logging.info(f"F1 {result}: {group_results[result]}")
            for class_name in gaps:

                gap = gaps[class_name]

                logging.info(f"Largest gap for {class_name} is between {gap[0]} and {gap[1]}: {gap[2]} ({gap[3]})")
                

def main(args):
    configpath = args.config

    #open config
    with open(configpath, "rb") as f:
        data = tomllib.load(f)

    randomseed = data["general"]["seed"]

    datasets = data["datasets"]

    #set randomness

    set_randomness(randomseed)

    # loop through all datasets provided in config
    for dataset in datasets:

        logging.info(f"Starting to build {dataset} dataset")

        # build one dataset to use for all models and all demonstration combinations
        train, test, test_labels, test_demographics = build_dataset(
            dataset, datasets[dataset]["path"]
        )

        logging.info(f"Built {dataset} dataset")

        demonstrations = datasets[dataset]['demonstrations']

        # loop through all demonstrations
        for demonstration in demonstrations:

            logging.info(f"Starting to create {demonstration} demonstration for {dataset} dataset")

            # create prompts from dataset
            prompts = build_demonstration(demonstration, demonstrations[demonstration], train, test)

            logging.info(f"Created {demonstration} demonstration for {dataset} dataset")

            # run dataset with all models provided
            run_dataset(
                prompts, test_labels, test_demographics, dataset, demonstration, datasets[dataset]['models']
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
    )

    args = parser.parse_args()

    main(args)
