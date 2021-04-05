import copy
import time
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from transformers import AutoModel


def compute_metrics(eval_pred):
    """Computes accuracy, f1, precision, and recall from a 
    transformers.trainer_utils.EvalPrediction object.
    """
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', pos_label=1)
    accuracy = accuracy_score(labels, preds)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, }

# Function to calculate the accuracy of our predictions vs labels


def metrics(preds, labels, total_metrics):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_flat, pred_flat, average='weighted', pos_label=1)
    accuracy = accuracy_score(labels_flat, pred_flat)
    list_metrics = ['accuracy', 'precision', 'recall', 'f1']
    cur_metrics = {'accuracy': accuracy,
                   'precision': precision, 'recall': recall, 'f1': f1}
    for m in list_metrics:
        total_metrics[m] += cur_metrics[m]
    return cur_metrics


def cost_function(cost_dict):
    return cost_dict['eval_loss']


def train(model, train_dataloader, cross_entropy, optimizer, device):

    model.train()

    total_loss = 0
    list_metrics = ['accuracy', 'precision', 'recall', 'f1']
    total_metrics = {}
    for m in list_metrics:
        total_metrics[m] = 0
    # empty list to save model predictions
    total_preds = []

    # iterate over batches
    for step, batch in enumerate(train_dataloader):

        # progress update after every 50 batches.
        """
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(
                step, len(train_dataloader)))
        """
        # push the batch to gpu
        #batch = [r.to(device) for r in batch]

        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        # clear previously calculated gradients
        model.zero_grad()

        # get model predictions for the current batch
        # potentially softmax

        preds = model(input_ids, mask)
        #soft = torch.nn.LogSoftmax(dim=-1)
        #max = soft(preds)
        # compute the loss between actual and predicted values
        loss = cross_entropy(preds, labels)

        # add on to the total loss
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # model predictions are stored on GPU. So, push it to CPU
        preds_a = preds.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        tmp_metrics = metrics(preds_a, label_ids,  total_metrics)
        #print(f'Tmp metrics {tmp_metrics}%')

        # append the model predictions
        total_preds.append(preds_a)

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)
    avg_metrics = {}
    for m in list_metrics:
        avg_metrics[m] = total_metrics[m] / len(train_dataloader)
    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    # returns the loss and predictions
    return avg_loss, total_preds, avg_metrics
# function for evaluating the model


def evaluate(model, dev_dataloader, cross_entropy, device):

    print("\nEvaluating...")

    # deactivate dropout layers
    model.eval()

    total_loss = 0
    list_metrics = ['accuracy', 'precision', 'recall', 'f1']
    total_metrics = {}
    for m in list_metrics:
        total_metrics[m] = 0

    # empty list to save the model predictions
    total_preds = []

    # iterate over batches
    for step, batch in enumerate(dev_dataloader):

        # Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:

            # Calculate elapsed time in minutes.
            #elapsed = format_time(time.time() - t0)

            # Report progress.
            """
            print('  Batch {:>5,}  of  {:>5,}.'.format(
                step, len(dev_dataloader)))
            """
        # push the batch to gpu
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # deactivate autograd
        with torch.no_grad():

            # model predictions
            preds = model(input_ids, mask)

            # compute the validation loss between actual and predicted values
            loss = cross_entropy(preds, labels)

            total_loss = total_loss + loss.item()

            preds_a = preds.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            tmp_metrics = metrics(preds_a, label_ids,  total_metrics)
            #print(f'Tmp metrics {tmp_metrics}%')
        total_preds.append(preds_a)

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(dev_dataloader)
    avg_metrics = {}
    for m in list_metrics:
        avg_metrics[m] = total_metrics[m] / len(dev_dataloader)
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds, avg_metrics


def model_init():
    """Returns an initialized model for use in a Hugging Face Trainer."""

    model = AutoModel.from_pretrained(
        'CAMeL-Lab/bert-base-camelbert-mix')
    return model
