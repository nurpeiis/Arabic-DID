import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar


def compare(x_list, y_list):
    """Compare two lists and returns a boolean list of matches between them.

    Args:
        x_list (`list`): A list of `str`.
        y_list (`list`): A list of `str`.

    Returns:
        `list`: A list of boolean values.
    """
    return list(np.char.equal(x_list, y_list))


def make_contingency_table(system_a, system_b, gold):
    """Make contingency table for McNemar test given predictions of two systems 
    and gold labels.

    Args:
        system_a (`list`): List of predictions from system A.
        system_b (`list`): List of predictions from system B.
        gold (`list`): List of gold labels.

    Returns:
        `list`: A contingency table.
    """
    tf_list_a = compare(system_a, gold)
    tf_list_b = compare(system_b, gold)

    both_correct = 0
    a_correct_b_incorrect = 0
    a_incorrect_b_correct = 0
    both_incorrect = 0

    for (a, b) in zip(tf_list_a, tf_list_b):
        if a == True and b == True:
            both_correct += 1
        elif a == True and b == False:
            a_correct_b_incorrect += 1
        elif a == False and b == True:
            a_incorrect_b_correct += 1
        elif a == False and b == False:
            both_incorrect += 1
        else:
            print('Something went wrong!')

    return [[both_correct, a_correct_b_incorrect],
            [a_incorrect_b_correct, both_incorrect]]


def mcnemar_pvalue(system_a, system_b, gold):
    """Calculate McNemar statitics.

    Args:
        system_a (`list`): List of predictions from system A.
        system_b (`list`): List of predictions from system b.
        gold (`list`): List of gold annotations.

    Returns:
        `float`: p-value
    """
    table = make_contingency_table(system_a, system_b, gold)
    return mcnemar(table).pvalue


def run_test(system_a, system_b, gold):
    # calculate p value
    pvalue = mcnemar_pvalue(system_a[1], system_b[1], gold)
    # run statitical analysis
    alpha = 0.05
    significance = 'NOT SIGNIFICANT' if pvalue > alpha else 'SIGNIFICANT'
    print(f'{system_a[0]}\t{system_b[0]}\t{significance}\t{pvalue}')

# TODO: country and region level


def file_run(file_a, file_b):
    df_a = pd.read_csv(file_a, sep='\t', header=0)
    df_b = pd.read_csv(file_b, sep='\t', header=0)
    gold = df_a['gold'].tolist()
    pred_a = df_a['pred'].tolist()
    pred_b = df_b['pred'].tolist()
    sys_a = [file_a, pred_a]
    sys_b = [file_b, pred_b]
    run_test(sys_a, sys_b, gold)


file_run('camel.tsv', 'camel_region.tsv')
file_run('camel.tsv', 'labels_test_salameh_agg.csv')
