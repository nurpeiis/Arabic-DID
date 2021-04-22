import torch
import random
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def metrics(preds, labels, total_metrics):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    f1 = f1_score(labels_flat, pred_flat, average='macro')
    precision = precision_score(labels_flat, pred_flat, average='macro')
    recall = recall_score(labels_flat, pred_flat, average='macro')
    accuracy = accuracy_score(labels_flat, pred_flat)

    list_metrics = ['accuracy', 'precision', 'recall', 'f1']
    cur_metrics = {'accuracy': accuracy,
                   'precision': precision, 'recall': recall, 'f1': f1}
    for m in list_metrics:
        total_metrics[m] += cur_metrics[m]
    return cur_metrics


def evaluate(model, dataloader, device):

    print("\nEvaluating...")
    model.to(device)
    # deactivate dropout layers
    model.eval()

    list_metrics = ['accuracy', 'precision', 'recall', 'f1', 'loss']
    total_metrics = {}
    for m in list_metrics:
        total_metrics[m] = 0

    # empty list to save the model predictions
    total_preds = None
    total_true_labels = None
    # iterate over batches
    for step, batch in enumerate(dataloader):

        # deactivate autograd
        with torch.no_grad():
            batch['input_ids'] = batch['input_ids'].to(device)
            batch['attention_mask'] = batch['attention_mask'].to(device)
            batch['token_type_ids'] = batch['token_type_ids'].to(device)
            batch['labels'] = batch['labels'].to(device)
            labels = batch['labels']
            outputs = model(**batch)
            # model outputs are always tuple in transformers (see doc)
            loss, logits = outputs[:2]

            total_metrics['loss'] += loss.item()

            preds_a = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()

            # print(f'Tmp metrics {tmp_metrics}%')
            if total_preds is None:
                total_preds = preds_a
                total_true_labels = label_ids
            else:
                total_preds = np.append(total_preds, preds_a, axis=0)
                total_true_labels = np.append(
                    total_true_labels,
                    label_ids,
                    axis=0)

    # compute avg metrics
    metrics(total_preds, total_true_labels,  total_metrics)
    # get argmax
    total_preds_argmax = np.argmax(total_preds, axis=1).flatten()

    return total_preds, total_preds_argmax, total_metrics
