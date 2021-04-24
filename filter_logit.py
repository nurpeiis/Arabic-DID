
import data_utils
import numpy as np


def get_space(aggregate_label_space_file, search_label_space_file):

    aggregate_labels, aggregate_label2id, aggregate_id2label = data_utils.get_label_space(
        aggregate_label_space_file)

    search_labels, search_label2id, search_id2label = data_utils.get_label_space(
        search_label_space_file)

    space = []
    for l in search_label2id.keys():
        if l in aggregate_label2id:
            space.append(aggregate_label2id[l])

    return space


def filter_logit(logit, space):

    pass


def share_weight(logit, space):
    # consolidating weights
    pass


def nullify_weight(logits, space, value_to_change=0.0):
    """Nullifies everything that is not in the space
    logit: a single logit, array of length num_labels
    space: array with indecies of the space of interest
    value_to_change: changes value of indecies that are not in space
    """
    rows = logits.shape[0]
    cols = logits.shape[1]
    logits = logits.tolist()
    for j in range(rows):
        for i in range(cols):
            if (i in space) == False:
                logits[j][i] = value_to_change

    # filtering
    return np.array(logits)
