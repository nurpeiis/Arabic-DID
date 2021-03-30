import torch


def encode_data(dataset, tokenizer, max_seq_length=128):
    """Featurizes the dataset into input IDs and attention masks for input into a
     transformer-style model.

     NOTE: This method should featurize the entire dataset simultaneously,
     rather than row-by-row.

  Args:
    dataset: A Pandas dataframe containing the data to be encoded.
    tokenizer: A transformers.PreTrainedTokenizerFast object that is used to
      tokenize the data.
    max_seq_length: Maximum sequence length to either pad or truncate every
      input example to.

  Returns:
    input_ids: A PyTorch.Tensor (with dimensions [len(dataset), max_seq_length])
      containing token IDs for the data.
    attention_mask: A PyTorch.Tensor (with dimensions [len(dataset), max_seq_length])
      containing attention masks for the data.
  """

    data = tokenizer(
        dataset['original_sentence'].tolist(),
        max_length=max_seq_length,
        padding='max_length',
        truncation=True
    )
    input_ids = torch.tensor(data['input_ids'], dtype=torch.long)
    attention_mask = torch.tensor(data['attention_mask'], dtype=torch.long)
    return input_ids, attention_mask


def extract_labels(dataset):
    """Converts labels into numerical labels.

  Args:
    dataset: A Pandas dataframe containing the labels in the column 'label'.

  Returns:
    labels: A list of integers corresponding to the labels for each example
  """
    labels = dataset['label'].tolist()
    for i in range(len(labels)):
        if labels[i] == True:
            labels[i] = 1
        else:
            labels[i] = 0

    return labels
