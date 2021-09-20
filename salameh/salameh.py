# -*- coding: utf-8 -*-

# MIT License
#
# Copyright 2018-2019 New York University Abu Dhabi
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


'''
'''


import collections
import os.path

import kenlm
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.metrics import precision_score
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.utils.dediac import dediac_ar
from utils import df2dialectsentence, levels_eval
import time
import json

ADIDA_LABELS = frozenset(['ALE', 'ALG', 'ALX', 'AMM', 'ASW', 'BAG', 'BAS',
                          'BEI', 'BEN', 'CAI', 'DAM', 'DOH', 'FES', 'JED',
                          'JER', 'KHA', 'MOS', 'MSA', 'MUS', 'RAB', 'RIY',
                          'SAL', 'SAN', 'SFX', 'TRI', 'TUN'])

ADIDA_LABELS_EXTRA = frozenset(['BEI', 'CAI', 'DOH', 'MSA', 'RAB', 'TUN'])
_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
_CHAR_LM_DIR = os.path.join(_DATA_DIR, 'lm', 'char')
_WORD_LM_DIR = os.path.join(_DATA_DIR, 'lm', 'word')

_TRAIN_DATA_AGGREGATED_PATH = os.path.join(
    _DATA_DIR, 'MADAR-Corpus-26-train.lines')

_TRAIN_DATA_PATH = os.path.join(_DATA_DIR, 'MADAR-Corpus-26-train.lines')
_TRAIN_DATA_EXTRA_PATH = os.path.join(_DATA_DIR, 'MADAR-Corpus-6-train.lines')
_VAL_DATA_PATH = os.path.join(_DATA_DIR, 'MADAR-Corpus-26-val.lines')
_TEST_DATA_PATH = os.path.join(_DATA_DIR, 'MADAR-Corpus-26-test.lines')


class DIDPred(collections.namedtuple('DIDPred', ['top', 'scores'])):
    """A named tuple containing dialect ID prediction results.
    Attributes:
        top (:obj:`str`): The dialect label with the highest score.
        scores (:obj:`dict`): A dictionary mapping each dialect label to it's
            computed score.
    """


class DialectIdError(Exception):
    """Base class for all CAMeL Dialect ID errors.
    """

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return str(self.msg)


class UntrainedModelError(DialectIdError):
    """Error thrown when attempting to use an untrained DialectIdentifier
    instance.
    """

    def __init__(self, msg):
        DialectIdError.__init__(self, msg)


class InvalidDataSetError(DialectIdError, ValueError):
    """Error thrown when an invalid data set name is given to eval.
    """

    def __init__(self, dataset):
        msg = ('Invalid data set name {}. Valid names are "TEST" and '
               '"VALIDATION"'.format(repr(dataset)))
        DialectIdError.__init__(self, msg)


def _normalize_lm_scores(scores):
    norm_scores = np.exp(scores)
    norm_scores = normalize(norm_scores)
    return norm_scores


def _word_to_char(txt):
    return ' '.join(list(txt.replace(' ', 'X')))


def _max_score(score_tups):
    max_score = -1
    max_dialect = None

    for dialect, score in score_tups:
        if score > max_score:
            max_score = score
            max_dialect = dialect

    return max_dialect


class DialectIdentifier(object):
    """A class for training, evaluating and running the dialect identification
    model described by Salameh et al. After initializing an instance, you must
    run the train method once before using it.
    Args:
        labels (set of str, optional): The set of dialect labels used in the
            training data in the main model.
            Defaults to ADIDA_LABELS.
        labels_extra (set of str, optional): The set of dialect labels used in
            the training data in the extra features model.
            Defaults to ADIDA_LABELS_EXTRA.
        char_lm_dir (str, optional): Path to the directory containing the
            character-based language models. If None, use the language models
            that come with this package.
            Defaults to None.
        word_lm_dir (str, optional): Path to the directory containing the
            word-based language models. If None, use the language models that
            come with this package.
            Defaults to None.
    """

    def __init__(self, labels=ADIDA_LABELS,
                 labels_extra=ADIDA_LABELS_EXTRA,
                 char_lm_dir=None,
                 word_lm_dir=None,
                 aggregated_layers=None,
                 result_file_name=None):
        if char_lm_dir is None:
            char_lm_dir = _CHAR_LM_DIR
        if word_lm_dir is None:
            word_lm_dir = _WORD_LM_DIR
        if result_file_name is None:
            result_file_name = time.strftime('%Y%m%d-%H%M%S.txt')

        # aggregated layer
        self.aggregated_layers = aggregated_layers
        self.result_file_name = result_file_name

        # salameh
        self._labels = labels
        self._labels_extra = labels_extra
        self._labels_sorted = sorted(labels)
        self._labels_extra_sorted = sorted(labels_extra)

        self._char_lms = collections.defaultdict(kenlm.Model)
        self._word_lms = collections.defaultdict(kenlm.Model)
        self._load_lms(char_lm_dir, word_lm_dir)

        self._is_trained = False

    def _load_lms(self, char_lm_dir, word_lm_dir):
        config = kenlm.Config()
        config.show_progress = False

        for label in self._labels:
            char_lm_path = os.path.join(char_lm_dir, '{}.arpa'.format(label))
            word_lm_path = os.path.join(word_lm_dir, '{}.arpa'.format(label))
            self._char_lms[label] = kenlm.Model(char_lm_path, config)
            self._word_lms[label] = kenlm.Model(word_lm_path, config)

    def _get_char_lm_scores(self, txt):
        chars = _word_to_char(txt)
        return np.array([self._char_lms[label].score(chars, bos=True, eos=True)
                         for label in self._labels_sorted])

    def _get_word_lm_scores(self, txt):
        return np.array([self._word_lms[label].score(txt, bos=True, eos=True)
                         for label in self._labels_sorted])

    def _get_lm_feats(self, txt):
        word_lm_scores = self._get_word_lm_scores(txt).reshape(1, -1)
        word_lm_scores = _normalize_lm_scores(word_lm_scores)
        char_lm_scores = self._get_char_lm_scores(txt).reshape(1, -1)
        char_lm_scores = _normalize_lm_scores(char_lm_scores)
        feats = np.concatenate((word_lm_scores, char_lm_scores), axis=1)
        return feats

    def _get_lm_feats_multi(self, sentences):
        feats_list = collections.deque()
        for sentence in sentences:
            feats_list.append(self._get_lm_feats(sentence))
        feats_matrix = np.array(feats_list)
        feats_matrix = feats_matrix.reshape((-1, 52))
        return feats_matrix

    def _prepare_sentences(self, sentences):

        # why use tokenization here, where in train we didn't use anything
        tokenized = [' '.join(simple_word_tokenize(dediac_ar(s)))
                     for s in sentences]
        sent_array = np.array(tokenized)
        x_trans = self._feat_union.transform(sent_array)
        x_trans_extra = self._feat_union_extra.transform(sent_array)
        x_predict_extra = self._classifier_extra.predict_proba(x_trans_extra)
        aggregated_prob_distrs = []
        aggregated_lm_feats = []
        # aggregated features
        if self.aggregated_layers:
            for i in range(len(self.aggregated_layers)):
                prob_distr, lm_feat = self.aggregated_layers[i].predict_proba_lm_feats(
                    sentences)
                aggregated_prob_distrs.append(prob_distr)
                aggregated_lm_feats.append(lm_feat)

        x_lm_feats = self._get_lm_feats_multi(sentences)
        x_final = sp.sparse.hstack(
            (x_trans, x_lm_feats, x_predict_extra))

        if self.aggregated_layers:
            for i in range(len(self.aggregated_layers)):
                if self.aggregated_layers[i].use_distr:
                    x_final = sp.sparse.hstack(
                        (x_final, aggregated_prob_distrs[i]))
                if self.aggregated_layers[i].use_lm:
                    x_final = sp.sparse.hstack(
                        (x_final, aggregated_lm_feats[i]))
        return x_final

    def train(self, data_path=None,
              data_extra_path=None,
              data_aggregated_path=None,
              level=None,
              char_ngram_range=(1, 3),
              word_ngram_range=(1, 1),
              n_jobs=None):
        """Trains the model on a given data set.
        Args:
            data_path (str, optional): Path to main training data. If None, use
                the provided training data.
                Defaults to None.
            data_extra_path (str, optional): Path to extra features training
                data. If None,cuse the provided training data.
                Defaults to None.
            char_ngram_range (tuple, optional): The n-gram ranges to consider
                in the character-based language models.
                Defaults to (1, 3).
            word_ngram_range (tuple, optional): The n-gram ranges to consider
                in the word-based language models.
                Defaults to (1, 1).
            n_jobs (int, optional): The number of parallel jobs to use for
                computation. If None, then only 1 job is used. If -1 then all
                processors are used.
                Defaults to None.
        """

        if data_path is None:
            data_path = _TRAIN_DATA_PATH
        if data_extra_path is None:
            data_extra_path = _TRAIN_DATA_EXTRA_PATH
        if data_aggregated_path is None:
            data_aggregated_path = _TRAIN_DATA_AGGREGATED_PATH
        if level is None:
            level = 'city'
        # Load training data and extract
        train_data = pd.read_csv(data_path, sep='\t', header=0)
        train_data_extra = pd.read_csv(data_extra_path, sep='\t', header=0)
        train_data_aggregated = pd.read_csv(
            data_aggregated_path, sep='\t', header=0)

        y, x = df2dialectsentence(train_data, level)
        y_extra, x_extra = df2dialectsentence(train_data_extra, level)

        # Build and train extra classifier
        print('Build and train extra classifier')
        self._label_encoder_extra = LabelEncoder()
        self._label_encoder_extra.fit(y_extra)
        y_trans = self._label_encoder_extra.transform(y_extra)

        word_vectorizer = TfidfVectorizer(lowercase=False,
                                          ngram_range=word_ngram_range,
                                          analyzer='word',
                                          tokenizer=lambda x: x.split(' '))
        char_vectorizer = TfidfVectorizer(lowercase=False,
                                          ngram_range=char_ngram_range,
                                          analyzer='char',
                                          tokenizer=lambda x: x.split(' '))
        self._feat_union_extra = FeatureUnion([('wordgrams', word_vectorizer),
                                               ('chargrams', char_vectorizer)])
        x_trans = self._feat_union_extra.fit_transform(x_extra)

        self._classifier_extra = OneVsRestClassifier(MultinomialNB(),
                                                     n_jobs=n_jobs)
        self._classifier_extra.fit(x_trans, y_trans)

        # Build and train aggreggated classifier
        print('Build and train aggreggated classifier')
        if self.aggregated_layers:
            for i in range(len(self.aggregated_layers)):
                self.aggregated_layers[i].train(train_data_aggregated)

        # Build and train main classifier
        print('Build and train main classifier')
        self._label_encoder = LabelEncoder()
        self._label_encoder.fit(y)
        y_trans = self._label_encoder.transform(y)

        word_vectorizer = TfidfVectorizer(lowercase=False,
                                          ngram_range=word_ngram_range,
                                          analyzer='word',
                                          tokenizer=lambda x: x.split(' '))
        char_vectorizer = TfidfVectorizer(lowercase=False,
                                          ngram_range=char_ngram_range,
                                          analyzer='char',
                                          tokenizer=lambda x: x.split(' '))
        self._feat_union = FeatureUnion([('wordgrams', word_vectorizer),
                                         ('chargrams', char_vectorizer)])
        self._feat_union.fit(x)

        x_prepared = self._prepare_sentences(x)

        self._classifier = OneVsRestClassifier(MultinomialNB(), n_jobs=n_jobs)
        self._classifier.fit(x_prepared, y_trans)

        self._is_trained = True

    def eval(self, data_path=None, data_set='VALIDATION', level=None):
        """Evaluate the trained model on a given data set.
        Args:
            data_path (str, optional): Path to an evaluation data set.
                If None, use one of the provided data sets instead.
                Defaults to None.
            data_set (str, optional): Name of the provided data set to use.
                This is ignored if data_path is not None. Can be either
                'VALIDATION' or 'TEST'. Defaults to 'VALIDATION'.
        Returns:
            dict: A dictionary mapping an evaluation metric to its computed
            value. The metrics used are accuracy, f1_micro, f1_macro,
            recall_micro, recall_macro, precision_micro and precision_macro.
        """

        if not self._is_trained:
            raise UntrainedModelError(
                'Can\'t evaluate an untrained model.')

        if data_path is None:
            if data_set == 'VALIDATION':
                data_path = _VAL_DATA_PATH
            elif data_set == 'TEST':
                data_path = _TEST_DATA_PATH
            else:
                raise InvalidDataSetError(data_set)
        if level is None:
            level = 'city'
        # Load eval data
        eval_data = pd.read_csv(data_path, sep='\t', header=0)
        y_true, x = df2dialectsentence(eval_data, level)

        # Generate predictions
        x_prepared = self._prepare_sentences(x)
        y_pred = self._classifier.predict(x_prepared)
        # print(self._classifier.predict_proba(x_prepared))
        y_pred = self._label_encoder.inverse_transform(y_pred)
        # Get scores
        levels_scores = levels_eval(y_true, y_pred, level)

        return levels_scores

    def record_experiment(self, scores):
        print(scores)
        final_record = {}
        final_record['scores'] = scores
        with open(self.result_file_name, 'w') as f:
            for i in range(len(self.aggregated_layers)):
                final_record[f'layer_{i}'] = self.aggregated_layers[i].dict_repr

            f.write(json.dumps(final_record))

        print('Recorded experiment in:', self.result_file_name)

    def predict(self, sentences):
        """Predict the dialect probability scores for a given list of
        sentences.
        Args:
            sentences (list of str): The list of sentences.
        Returns:
            list of DIDPred: A list of prediction results, each corresponding
                to its respective sentence.
        """

        if not self._is_trained:
            raise UntrainedModelError(
                'Can\'t predict with an untrained model.')

        x_prepared = self._prepare_sentences(sentences)
        predicted_scores = self._classifier.predict_proba(x_prepared)
        result = collections.deque()
        for sentence, scores in zip(sentences, predicted_scores):
            score_tups = list(zip(self._labels_sorted, scores))
            predicted_dialect = _max_score(score_tups)
            dialect_scores = dict(score_tups)
            result.append(DIDPred(predicted_dialect, dialect_scores))

        return list(result)


if __name__ == '__main__':
    d = DialectIdentifier()
    d.train()
    print(d.eval(data_set='TEST'))
