import data_utils
import torch

from torch.utils.data import Dataset


class DIDDataset(Dataset):
    """
    A torch.utils.data.Dataset wrapper for the DID.
    """

    def __init__(self, dataframe, tokenizer, level, label_space_file='',  max_seq_length=256):
        """
        Args:
          dataframe: A Pandas dataframe containing the data.
          tokenizer: A transformers.PreTrainedTokenizerFast object that is used to
            tokenize the data.
          level: A level at which the labels are
          label_space_file: A file for label space, where dialect converted to integer
          max_seq_length: Maximum sequence length to either pad or truncate every
            input example to.
        """
        self.encoded_data = data_utils.encode_data(
            dataframe, tokenizer, max_seq_length)
        self.label_list = data_utils.extract_labels(
            dataframe, level, label_space_file)

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, i):
        """
        Returns:
          example: A dictionary containing the input_ids, attention_mask, and
            label for the i-th example, with the values being numeric tensors
            and the keys being 'input_ids', 'attention_mask', and 'labels'.
        """
        print('here')
        return {'input_ids': self.encoded_data[0][i], 'attention_mask':  self.encoded_data[1][i], 'labels': torch.tensor(self.label_list[i], dtype=torch.long)}
