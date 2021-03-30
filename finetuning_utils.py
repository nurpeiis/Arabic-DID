
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from transformers import AutoModel


def compute_metrics(eval_pred):
    """Computes accuracy, f1, precision, and recall from a 
    transformers.trainer_utils.EvalPrediction object.
    """
    # TODO: Return a dictionary containing the accuracy, f1, precision, and recall scores.
    # You may use sklearn's precision_recall_fscore_support and accuracy_score methods.

    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', pos_label=1)
    accuracy = accuracy_score(labels, preds)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, }


def cost_function(cost_dict):
    return cost_dict['eval_loss']


def model_init():
    """Returns an initialized model for use in a Hugging Face Trainer."""

    model = AutoModel.from_pretrained('CAMeL-Lab/bert-base-camelbert-mix')
    return model
