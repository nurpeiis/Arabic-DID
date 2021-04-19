# -*- coding: utf-8 -*-

# MIT License
#
# Copyright 2018-2021 New York University Abu Dhabi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Text classification tasks utils
"""

import logging
import os
import re
import data_utils
import camel_tools.utils.normalize as normalize
import camel_tools.utils.dediac as dediac
from transformers.data.processors.utils import (
    DataProcessor,
    InputExample,
    InputFeatures
)

logger = logging.getLogger(__name__)


def convert_examples_to_features(examples, tokenizer,
                                 max_length=512,
                                 task=None,
                                 label_list=None,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """
    Loads a data file into a list of ``InputFeatures``
    Args:
        examples: List of ``InputExamples`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: Arabic Sentiment Analysis task
        label_list: List of labels.
                    Can be obtained from the processor using the
                    ``processor.get_labels()`` method
        pad_on_left: If set to ``True``,
                     the examples will be padded on the left rather
                     than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token
                              (It is usually 0, but can vary such as for
                               XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``,
                                the attention mask will be filled by ``1``
                                for actual values
                                and by ``0`` for padded values.
                                If set to ``False``, inverts it (``1`` for
                                padded values, ``0`` for actual values)
    Returns:
        list of task-specific ``InputFeatures`` which can be fed to the model.
    """

    label_map = {label: i for i, label in enumerate(label_list)}
    logger.info('**LABEL MAP**')
    logger.info(label_map)
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))

        inputs = tokenizer.encode_plus(
            example.text_a,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1]
                              * padding_length) + attention_mask
            token_type_ids = (([pad_token_segment_id] * padding_length)
                              + token_type_ids)
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero
                                                else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id]
                                               * padding_length)

        assert len(input_ids) == max_length, (
            "Error with input length {} vs {}".format(len(input_ids),
                                                      max_length))

        assert len(attention_mask) == max_length, (
            "Error with input length {} vs {}".format(len(attention_mask),
                                                      max_length))

        assert len(token_type_ids) == max_length, (
            "Error with input length {} vs {}".format(len(token_type_ids),
                                                      max_length))

        label = (label_map[example.label] if example.label is not None
                 else label_map['ms'])

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" %
                        " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" %
                        " ".join([str(x) for x in token_type_ids]))

            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label=label))

    return features


class DIDProcesser(DataProcessor):

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['text'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def _create_examples(self, df, level, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        if level == 'city':
            labels = df[['dialect_city_id', 'dialect_country_id',
                         'dialect_region_id']].values.tolist()
        elif level == 'country':
            labels = df[['dialect_country_id',
                         'dialect_region_id']].values.tolist()
        elif level == 'region':
            labels = df[['dialect_region_id']].values.tolist()

        labels = [' '.join(label) for label in labels]
        sentences = df['original_sentence'].tolist()

        for (i, sentence) in enumerate(sentences):
            guid = "%s-%s" % (set_type, i)
            text_a = self.process_text(sentence)
            label = labels[i]
            examples.append(InputExample(
                guid=guid, text_a=text_a, label=label))

        return examples

    def get_examples(self, files, level, set_type):
        """See base class."""
        logger.info(f'Looking at files: {files}')
        df = data_utils.get_df_from_files(files)
        return self._create_examples(df, level, set_type)

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


output_modes = {
    "arabic_did_madar_26": "classification"
}
