
import os


class LayerObject:

    def __init__(self, labels, data_dir, train_path, test_path, val_path):
        self.labels = labels
        self.data_dir = os.path.join(os.path.dirname(__file__), data_dir)
        self.char_lm_dir = os.path.join(data_dir, 'lm', 'char')
        self.word_lm_dir = os.path.join(data_dir, 'lm', 'word')
        self.train_path = train_path
        self.test_path = test_path
        self.val_path = val_path
