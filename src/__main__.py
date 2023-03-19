import argparse

import logging

import os

import tomllib

from typing import List, Dict, Any

import random

import numpy as np

from sklearn.metrics import confusion_matrix, f1_score

from .datasets.biasinbios import BiasInBios
from .datasets.twitteraae import TwitterAAE
from .datasets.hatexplain import HatExplain

from .models.gpt import GPT
from .models.chatgpt import ChatGPT
from .models.hf import HF

from .demonstrations.random import RandomSampler

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def set_randomness(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def metrics(responses: List[str], labels: List[str], dataset: str, demographics: List[List[str]]):

    demographic_groups = None
    
    if dataset == "hatexplain-race":
        demographic_groups = ["African", "Arab", "Asian", "Hispanic", "Caucasian", "Indian", "Indigenous"]
    
    elif dataset == "hateexplain-gender":
        demographic_groups = ["Men", "Women"]
    
    else:
        demographic_groups = list(set([element for sublist in demographics for element in sublist]))
    
    labels_set = list(set(labels))

    labels_dict = dict(zip(labels_set, range(len(labels_set))))

    dummy_labels = [labels_dict[x] for x in labels]

    dummy_responses = []

    for response in responses:
        for label in labels_set:
            if response.find(label) != -1:
                dummy_responses.append(labels_dict[label])
                break
        
        else:
            dummy_responses.append(-1)
    
    dummy_responses = np.array(dummy_responses)
    dummy_labels = np.array(dummy_labels)

    F1_Overall = f1_score(dummy_labels, dummy_responses, average='macro')

    f1_per_group = dict()

    tpr_per_group = dict()
    
    for demographic_group in demographic_groups:

        indicies = [index for index, item in enumerate(demographics) if demographic_group in item]

        cnf_matrix = confusion_matrix(dummy_labels[indicies], dummy_responses[indicies])

        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)

        FN = FN.astype(float)
        TP = TP.astype(float)

        f1_per_group[demographic_group] = f1_score(dummy_labels[indicies], dummy_responses[indicies])

        tpr_per_group[demographic_group] = TP/(TP+FN)
    
    gaps = []

    for group1 in tpr_per_group:
        for group2 in tpr_per_group:
            gap =  tpr_per_group[group1] - tpr_per_group[group2]
            one_minus_gap = 1-gap
            gaps.append((group1, group2, gap, one_minus_gap))
    
    gaps = sorted(gaps, key=lambda x: x[2], reversed = True)

    results = {
        "F1 Overall": F1_Overall,
        "F1 Per Group" : f1_per_group,
        "GAP" : gaps[0]
    }

    return results
    

def build_demonstration(
    demonstration_name: str,
    demonstration_params: Dict[str, Any],
    train: List[str],
    test: List[str],
) -> List[str]:
    shots = None

    if demonstration_name == "zeroshot":
        shots = 0
    elif demonstration_name == "random":
        shots = demonstration_params["shots"]
    else:
        raise ValueError(f"{demonstration_name} does not exist!")

    sampler = RandomSampler(shots=shots)

    return sampler.create_demonstrations(train, test)


def build_model(model_name: str):
    models = {
        "gpt3": GPT("text-curie-001"),
        "chatgpt": ChatGPT("gpt-3.5-turbo"),
        "flan": HF("https://api-inference.huggingface.co/models/google/flan-t5-large"),
        "ul2": HF("https://api-inference.huggingface.co/models/google/ul2"),
    }

    model = None

    try:
        model = models[model_name]
    except KeyError:
        raise ValueError(f"{model_name} does not exist!")

    return model


def build_dataset(dataset_name: str, path: str):
    datasets = {
        "hatexplain-gender": HatExplain,
        "hatexplain-race": HatExplain,
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

        model = build_model(model_name)

        logging.info(f"Created {model_name} model for {dataset} with {demonstration}")

        logging.info(f"Running {model_name} on {dataset} with {demonstration}")

        responses = model.generate_from_prompts(prompts)

        if not os.path.exists("./output/responses"):
            os.makedirs("./output/responses", exist_ok=True)

        with open(os.path.join("./output/responses", f"{model_name}_{dataset}_{demonstration}"), "w") as f:
            for response in responses:
                f.write(response)

        logging.info(f"Completed running {model_name} on {dataset} with {demonstration}")

        logging.info(f"Calculating metrics for {model_name} on {dataset} with {demonstration}")

        result = metrics(responses, test_labels, dataset, test_demographics)

        gap = result['GAP']

        group_results = result['F1 Per Group']

        logging.info(f"For {model_name} on {dataset}")
        logging.info(f"F1 Overall: {result['F1 Overall']}")
        logging.info(f"Largest Gap between: {gap[0]} and {gap[1]} is {gap[2]} ({gap[3]})")
        for result in group_results:
            logging.info(f"F1 {result}: {group_results[result]}")
                

def main(args):
    configpath = args.config

    with open(configpath, "rb") as f:
        data = tomllib.load(f)

    randomseed = data["general"]["seed"]

    datasets = data["datasets"]

    set_randomness(randomseed)

    for dataset in datasets:

        logging.info(f"Starting to build {dataset} dataset")

        train, test, test_labels, test_demographics = build_dataset(
            dataset, datasets[dataset]["path"]
        )

        logging.info(f"Built {dataset} dataset")

        demonstrations = datasets[dataset]['demonstrations']

        for demonstration in demonstrations:

            logging.info(f"Starting to create {demonstration} demonstration for {dataset} dataset")

            prompts = build_demonstration(demonstration, demonstrations[demonstration], train, test)

            logging.info(f"Created {demonstration} demonstration for {dataset} dataset")

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
