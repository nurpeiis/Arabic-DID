import torch.nn as nn


class Classifier(nn.Module):

    def __init__(self, bert, dropout_prob, hidden_size, num_classes):

        super(Classifier, self).__init__()

        # bert layer
        self.bert = bert

        # dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # relu activation function
        self.relu = nn.ReLU()

        # dense layer  (Output layer)
        self.fc = nn.Linear(hidden_size, num_classes)

    # define the forward pass

    def forward(self, input_ids, mask):

        # pass the inputs to the model
        out = self.bert(input_ids, attention_mask=mask)

        x = out[1]

        x = self.relu(x)

        x = self.dropout(x)

        # output layer
        logits = self.fc(x)

        return logits
