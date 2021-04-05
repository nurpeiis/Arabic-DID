import torch.nn as nn


class Classifier(nn.Module):

    def __init__(self, bert, dropout_prob, hidden_size1, hidden_size2, num_classes):

        super(Classifier, self).__init__()

        self.bert = bert

        # dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # relu activation function
        self.relu = nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(hidden_size1, hidden_size2)

        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(hidden_size2, num_classes)

    # define the forward pass

    def forward(self, input_ids, mask):

        # pass the inputs to the model
        out = self.bert(input_ids, attention_mask=mask)

        x = self.fc1(out[1])

        x = self.relu(x)

        x = self.dropout(x)

        # output layer
        logits = self.fc2(x)

        return logits
