from typing import List, Any, Dict

import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

import copy


def metrics(
    preds: List[str],
    labels: List[str],
    dataset: str,
    demographics: List[List[str]],
    overall_demographics: List[str],
) -> Dict[str, Any]:
    """Returns a dictionary of overall macro f1 score,
    recall, specificity, and macro f1 for each group
    and largest gap in recall for each label

    Code heavily modified from https://github.com/pocaguirre/jiant/blob/master/jiant/ext/fairness/DF_training.py

    :param preds: list of predictions from model
    :type preds: List[str]
    :param labels: list of groundtruth labels
    :type labels: List[str]
    :param dataset: name of dataset
    :type dataset: str
    :param demographics: list of lists of demographic groups associated with labels
    :type demographics: List[List[str]]
    :param overall_demographics: Demographics to focus on in the output
    :type overall_demographics: List[str]
    :return: dictionary of overall macro f1 score, recall, specificity, and macro f1 for each group
    and largest gap in recall for each label
    :rtype: Dict[str, Any]
    """

    scores = {"recall": {}, "specificity": {}, "score": {}}

    # create subgroups to focus on
    preds_clean = copy.deepcopy(preds)

    # clean up predictions
    preds_clean = [x.lower() for x in preds_clean]

    conv = lambda i: i or ""
    preds_clean = [conv(i) for i in preds_clean]

    if dataset == "bias":
        preds_clean = [x.replace("lawyer", "attorney") for x in preds_clean]

    # create list of all labels
    labels_set = list(set(labels))

    # map labels to numbers to make it easier for sklearn calculations
    labels_dict = dict(zip(labels_set, range(len(labels_set))))

    # map the labels lists to dummy labels
    dummy_labels = [labels_dict[x] for x in labels]

    dummy_preds = []

    for pred in preds_clean:
        # see if any of the labels are in the response
        for label in labels_set:
            if pred.find(label) != -1:
                dummy_preds.append(labels_dict[label])
                break
            # if not we add -1 instead
        else:
            dummy_preds.append(-1)

    dummy_preds = np.array(dummy_preds)
    dummy_labels = np.array(dummy_labels)

    # remove predictions that have demographics not in the set
    # mostly for hatexplain which has multiple demographics per label
    demographic_index = [
        i
        for i, item in enumerate(demographics)
        if len(set(overall_demographics).intersection(set(item))) != 0
    ]

    demographics_filtered = copy.deepcopy([demographics[i] for i in demographic_index])

    dummy_preds = dummy_preds[demographic_index]
    dummy_labels = dummy_labels[demographic_index]

    # get total score
    scores["total_score"] = f1_score(
        dummy_labels, dummy_preds, average="macro", labels=list(labels_dict.values())
    )

    for dem in overall_demographics:
        # filter out items that do not have the specified demographic
        index = [i for i, item in enumerate(demographics_filtered) if dem in item]

        # calculate f1, recall and specifity for those items
        cnf_matrix = confusion_matrix(
            dummy_labels[index], dummy_preds[index], labels=list(labels_dict.values())
        )

        fn = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        tp = np.diag(cnf_matrix)
        fp = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
        tn = cnf_matrix.sum() - (fp + fn + tp)

        fn = fn.astype(float)
        tp = tp.astype(float)
        fp = fp.astype(float)
        tn = tn.astype(float)

        score = f1_score(
            dummy_labels[index],
            dummy_preds[index],
            average="macro",
            labels=list(labels_dict.values()),
        )

        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)

        scores["recall"][dem] = recall
        scores["specificity"][dem] = specificity
        scores["score"][dem] = score

    gaps = []

    # calculate all the TPR gaps for every possible combination of demographics
    for group1 in scores["recall"]:
        for group2 in scores["recall"]:
            gap = scores["recall"][group1] - scores["recall"][group2]

            gap = np.nan_to_num(gap)

            one_minus_gap = 1 - gap
            gaps.append([group1, group2, gap, one_minus_gap])

    # get the maximum TPR gap per class
    max_gaps = dict()
    for i, label in enumerate(labels_set):
        gaps = sorted(gaps, key=lambda x: x[2][i], reverse=True)

        max_gaps[label] = copy.deepcopy(gaps[0])
        max_gaps[label][2] = max_gaps[label][2][i]
        max_gaps[label][3] = max_gaps[label][3][i]

    scores["max_gaps"] = max_gaps

    return scores
