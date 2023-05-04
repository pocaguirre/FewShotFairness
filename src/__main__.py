import argparse

import csv

import logging

import os

import tomli

from typing import List, Dict, Any, Tuple

import random

import numpy as np
import pandas as pd

from .datasets import *

from .models import *

from .demonstrations import *

from .utils import metrics

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


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
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    overall_demographics: List[str],
) -> Tuple[List[str], pd.DataFrame, str]:
    """Build demonstrations based on parameters

    :param demonstration_name: name of demonstration
    :type demonstration_name: str
    :param demonstration_params: parameters for demonstration
    :type demonstration_params: Dict[str, Any]
    :param train_df: train prompts and demographics
    :type train_df: pd.DataFrame
    :param test_df: train prompts and demographics
    :type test_df: pd.DataFrame
    :param overall_demographics: demographics to focus on
    :type overall_demographics: List[str]
    :raises ValueError: demonstration does not exist
    :return: the list of formed demonstrations
    :rtype: List[str]
    """

    demonstrations = {
        "excluding": ExcludingDemographic,
        "zeroshot": RandomSampler,
        "random": RandomSampler,
        "stratified": StratifiedSampler,
        "within": WithinDemographic,
        "similarity": SimilarityDemonstration,
        "diversity": DiversityDemonstration,
    }

    shots = None

    if demonstration_name == "zeroshot":
        shots = 0
        demonstration_params["shots"] = 0
    else:
        shots = demonstration_params["shots"]

    try:
        demonstration = demonstrations[demonstration_name]
    except KeyError:
        raise ValueError(f"{demonstration_name} does not exist!")

    sampler = demonstration(shots=shots)

    prompts, filtered_test_df = sampler.create_demonstrations(
        train_df, test_df, overall_demographics
    )

    return prompts, filtered_test_df, sampler.type


def build_model(model_name: str, model_params: Dict[str, Any]) -> apimodel:
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
        "gpt3": ("gpt", "text-davinci-003"),
        "davinci-002": ("gpt", "text-davinci-002"),
        "chatgpt":  ("chatgpt", "gpt-3.5-turbo"),
        "flan-ul2": ("hfoffline", "google/flan-ul2"),
        "ul2": ("hf", "https://api-inference.huggingface.co/models/google/ul2"),
        "offline-ul2": ("hfoffline", "google/ul2"),
        "alpaca-13b" : ("hfoffline", "chavinlo/alpaca-13b"),
        "alpaca-65b": ("hfoffline", "chavinlo/alpaca-native"),
        "llama-13b": ("hfoffline", "decapoda-research/llama-13b-hf"),
        "llama-65b" : ("hfoffline", "decapoda-research/llama-65b-hf"),
    }

    class_ = None

    try:
        model_info = models[model_name]
        class_ = globals()[model_info[0]]
    except KeyError:
        raise ValueError(f"model {model_name} does not exit")

    instance = class_(model_info[1], **model_params)

    return instance
   

def build_dataset(
    dataset_name: str, path: str
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Build dataset based on name and path to it

    :param dataset_name: name of dataset
    :type dataset_name: str
    :param path: path of dataset
    :type path: str
    :raises ValueError: dataset name does not exist
    :return: Created dataset
    :rtype: Tuple[pd.DataFrame, pd.DataFrame, List[str]]
    """
    datasets = {
        "hatexplain-race": HateXplainRace,
        "hatexplain-gender": HateXplainGender,
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
    test_df: pd.DataFrame,
    overall_demographics: List[str],
    dataset: str,
    demonstration: str,
    demonstration_type: str,
    demonstration_params: Dict[str, Any],
    models: List[str],
    output_folder: str,
):
    results = []

    for model_name in models:
        logging.info(
            f"Starting to create {model_name} model for {dataset} with {demonstration}"
        )

        model = build_model(model_name, models[model_name])

        logging.info(f"Created {model_name} model for {dataset} with {demonstration}")

        logging.info(f"Running {model_name} on {dataset} with {demonstration}")

        responses = model.generate_from_prompts(prompts)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)

        with open(
            os.path.join(output_folder, f"{model_name}_{dataset}_{demonstration}.csv"),
            "w",
        ) as csvfile:
            csvwriter = csv.writer(csvfile)

            csvwriter.writerow(["prompt", "response", "label", "demographic"])

            for prompt, response, label, demographic in zip(
                prompts,
                responses,
                test_df["labels"].tolist(),
                test_df["demographics"].tolist(),
            ):
                csvwriter.writerow([prompt, response, label, demographic])

        logging.info(
            f"Completed running {model_name} on {dataset} with {demonstration}"
        )

        logging.info(
            f"Calculating metrics for {model_name} on {dataset} with {demonstration}"
        )

        performance = metrics(
            responses,
            test_df["labels"].tolist(),
            dataset,
            test_df["demographics"].tolist(),
            overall_demographics,
        )

        result = [
            model_name,
            demonstration_params["shots"],
            demonstration_type,
            demonstration,
            performance["total_score"],
        ]

        group_results = performance["score"]

        recall_results = performance["recall"]

        gaps = performance["max_gaps"]

        for group_result in group_results:
            result.append({group_result: group_results[group_result]})

        gaps = dict(sorted(gaps.items(), key=lambda item: item[1][3]))

        result.append(list(gaps.values())[0][3])

        for class_name in gaps:
            gap = gaps[class_name]

            result.append({class_name: list(gap)})
        
        for recall_result in recall_results:
             result.append({recall_result: recall_results[recall_result]})

        results.append(result)

    if not os.path.exists(os.path.join(output_folder, "results")):
        os.makedirs(os.path.join(output_folder, "results"), exist_ok=True)

    with open(
        os.path.join(
            output_folder, "results", f"results_{dataset}_{demonstration}.csv"
        ),
        "w",
    ) as csvfile:
        csvwriter = csv.writer(csvfile)

        for prepared_result in results:
            csvwriter.writerow(prepared_result)


def main(args):
    configpath = args.config

    # open config
    with open(configpath, "rb") as f:
        data = tomli.load(f)

    randomseed = data["general"]["seed"]

    output_folder = data["general"]["output_folder"]

    datasets = data["datasets"]

    # set randomness

    set_randomness(randomseed)

    # loop through all datasets provided in config
    for dataset in datasets:
        logging.info(f"Starting to build {dataset} dataset")

        # build one dataset to use for all models and all demonstration combinations
        train_df, test_df, overall_demographics = build_dataset(
            dataset, datasets[dataset]["path"]
        )

        logging.info(f"Built {dataset} dataset")

        demonstrations = datasets[dataset]["demonstrations"]

        # loop through all demonstrations
        for demonstration in demonstrations:
            logging.info(
                f"Starting to create {demonstration} demonstration for {dataset} dataset"
            )

            # create prompts from dataset
            prompts, filtered_test_df, demonstration_type = build_demonstration(
                demonstration,
                demonstrations[demonstration],
                train_df,
                test_df,
                overall_demographics,
            )

            logging.info(f"Created {demonstration} demonstration for {dataset} dataset")

            # run dataset with all models provided
            run_dataset(
                prompts,
                filtered_test_df,
                overall_demographics,
                dataset,
                demonstration,
                demonstration_type,
                demonstrations[demonstration],
                datasets[dataset]["models"],
                output_folder,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)

    args = parser.parse_args()

    main(args)
