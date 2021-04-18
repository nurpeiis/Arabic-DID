import torch
import pandas as pd


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


def get_df_from_files(files):
    if len(files) == 0:
        return pd.DataFrame()
    df = pd.read_csv(
        files[0], sep='\t', header=0)
    for i in range(1, len(files)):
        df = df.append(pd.read_csv(
            files[i], sep='\t', header=0), ignore_index=True)

    return df


def process_text(self, text):
    """
    processes the input text by removing diacritics
    Args:
        input text
    Returns:
        processed text
    """

    text = dediac.dediac_ar(text)
    return text


def extract_labels(dataset, level, label_space_file=''):
    """Converts labels into numerical labels.

    Args:
      dataset: A Pandas dataframe containing the labels in the column 'label'.
      level: A level at which the labels are
      label_space_file: A file for label space, where dialect converted to integer
    Returns:
      labels: A list of integers corresponding to the labels for each example
    """

    labels = dataset[f'dialect_{level}_id'].tolist()
    dictionary_label_to_index = dict()
    if label_space_file == '':
        label_space_file = f'labels/{level}_label_id.txt'
    with open(label_space_file, 'r') as f:
        lines = f.readlines()
        for l in lines:
            dictionary_label_to_index[l.split(',')[0]] = int(
                l.split(',')[1][:-1])

    for i in range(len(labels)):
        if labels[i] in dictionary_label_to_index:
            labels[i] = dictionary_label_to_index[labels[i]]
        else:
            labels[i] = 0
    return labels
