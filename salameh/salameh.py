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
from utils import LayerObject

ADIDA_LABELS = frozenset(['ALE', 'ALG', 'ALX', 'AMM', 'ASW', 'BAG', 'BAS',
                          'BEI', 'BEN', 'CAI', 'DAM', 'DOH', 'FES', 'JED',
                          'JER', 'KHA', 'MOS', 'MSA', 'MUS', 'RAB', 'RIY',
                          'SAL', 'SAN', 'SFX', 'TRI', 'TUN'])

ADIDA_LABELS_EXTRA = frozenset(['BEI', 'CAI', 'DOH', 'MSA', 'RAB', 'TUN'])
# AGGREGATED_LABELS_EXTRA = frozenset(['amarah-iq-gulf', 'shibin_el_kom-eg-nile_basin', 'muscat-om-gulf', 'asyut-eg-nile_basin', 'kut-iq-gulf', 'luxor-eg-nile_basin', 'hail-sa-gulf', 'kafr_el_sheikh-eg-nile_basin', 'marrakesh-ma-maghreb', 'aqaba-jo-levant', 'meknes-ma-maghreb', 'amman-jo-levant', 'bayda-ly-maghreb', 'halba-lb-levant', 'umm_al_quwain-ae-gulf', 'najaf-iq-gulf', 'hawalli-kw-gulf', 'sidon-lb-levant', 'khartoum-sd-nile_basin', 'tripoli-ly-maghreb', 'tobruk-ly-maghreb', 'annaba-dz-maghreb', 'msa-msa-msa', 'bordj_bou_arreridj-dz-maghreb', 'jijel-dz-maghreb', 'abu_dhabi-ae-gulf', 'fes-ma-maghreb', 'aleppo-sy-levant', 'suez-eg-nile_basin', 'ismailia-eg-nile_basin', 'samawah-iq-gulf', 'doha-qa-gulf', 'mansoura-eg-nile_basin', 'damascus-sy-levant', 'al_rayyan-qa-gulf', 'girga-eg-nile_basin', 'cairo-eg-nile_basin', 'buraidah-sa-gulf', 'riyadh-sa-gulf', 'karbala-iq-gulf', 'duhok-iq-gulf', 'el_arish-eg-nile_basin', 'oujda-ma-maghreb', 'aswan-eg-nile_basin', 'manama-bh-gulf', 'oran-dz-maghreb', 'jizan-sa-gulf', 'mahdia-tn-maghreb', 'jeddah-sa-gulf', 'agadir-ma-maghreb', 'beirut-lb-levant', 'tripoli-lb-levant', 'al_suwayda-sy-levant', 'tanta-eg-nile_basin', 'dammam-sa-gulf', 'mogadishu-so-gulf_aden',
# 'sfax-tn-maghreb', 'salalah-om-gulf', 'al_hudaydah-ye-gulf_aden', 'hurghada-eg-nile_basin', 'basra-iq-gulf', 'zagazig-eg-nile_basin', 'salt-jo-levant', 'rabat-ma-maghreb', 'sohar-om-gulf', 'abha-sa-gulf', 'fujairah-ae-gulf', 'mosul-iq-gulf', 'baghdad-iq-gulf', 'ariana-tn-maghreb', 'el_tor-eg-nile_basin', 'homs-sy-levant', 'beni_suef-eg-nile_basin', 'najran-sa-gulf', 'ramadi-iq-gulf', 'faiyum-eg-nile_basin', 'ouargla-dz-maghreb', 'ras_al_khaimah-ae-gulf', 'algiers-dz-maghreb', 'nouakchott-mr-maghreb', 'tabuk-sa-gulf', 'tunis-tn-maghreb', 'minya-eg-nile_basin', 'dhamar-ye-gulf_aden', 'sousse-tn-maghreb', 'erbil-iq-gulf', 'khasab-om-gulf', 'bouira-dz-maghreb', 'djibouti-dj-gulf_aden', 'ibb-ye-gulf_aden', 'al_madinah-sa-gulf', 'jerusalem-ps-levant', 'khenchela-dz-maghreb', 'qena-eg-nile_basin', 'jahra-kw-gulf', 'kairouan-tn-maghreb', 'damanhur-eg-nile_basin', 'alexandria-eg-nile_basin', 'port_said-eg-nile_basin', 'sanaa-ye-gulf_aden', 'dubai-ae-gulf', 'giza-eg-nile_basin', 'sulaymaniyah-iq-gulf', 'latakia-sy-levant', 'zarqa-jo-levant', 'sur-om-gulf', 'nizwa-om-gulf', 'aden-ye-gulf_aden', 'benghazi-ly-maghreb', 'bechar-dz-maghreb', 'gaza-ps-levant', 'misrata-ly-maghreb', 'tangier-ma-maghreb'])
AGGREGATED_LABELS_EXTRA = frozenset(['ma-maghreb', 'bh-gulf', 'msa-msa', 'tn-maghreb', 'sd-nile_basin', 'dj-gulf_aden', 'mr-maghreb', 'ps-levant', 'lb-levant', 'eg-nile_basin',
                                    'ye-gulf_aden', 'gulf-gulf', 'dz-maghreb', 'iq-gulf', 'ps,jo-levant', 'qa-gulf', 'jo-levant', 'so-gulf_aden', 'ae-gulf', 'om-gulf', 'ly-maghreb', 'kw-gulf', 'sa-gulf', 'sy-levant'])
_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
_CHAR_LM_DIR = os.path.join(_DATA_DIR, 'lm', 'char')
_WORD_LM_DIR = os.path.join(_DATA_DIR, 'lm', 'word')
_AGGREGATED_DIR = os.path.join(os.path.dirname(__file__), 'aggregated_country')
_AGGREGATED_CHAR_LM_DIR = os.path.join(_AGGREGATED_DIR, 'lm', 'char')
_AGGREGATED_WORD_LM_DIR = os.path.join(_AGGREGATED_DIR, 'lm', 'word')
_TRAIN_DATA_AGGREGATED_PATH = os.path.join(
    _AGGREGATED_DIR, 'MADAR-Corpus-26-train.lines')
_TRAIN_DATA_PATH = os.path.join(_DATA_DIR, 'corpus_26_train.tsv')
_TRAIN_DATA_EXTRA_PATH = os.path.join(_DATA_DIR, 'corpus_6_train.tsv')
_VAL_DATA_PATH = os.path.join(_DATA_DIR, 'corpus_26_val.tsv')
_TEST_DATA_PATH = os.path.join(_DATA_DIR, 'corpus_26_test.tsv')


city_layer = LayerObject(frozenset(['amarah-iq-gulf', 'shibin_el_kom-eg-nile_basin', 'muscat-om-gulf', 'asyut-eg-nile_basin', 'kut-iq-gulf', 'luxor-eg-nile_basin', 'hail-sa-gulf', 'kafr_el_sheikh-eg-nile_basin', 'marrakesh-ma-maghreb', 'aqaba-jo-levant', 'meknes-ma-maghreb', 'amman-jo-levant', 'bayda-ly-maghreb', 'halba-lb-levant', 'umm_al_quwain-ae-gulf', 'najaf-iq-gulf', 'hawalli-kw-gulf', 'sidon-lb-levant', 'khartoum-sd-nile_basin', 'tripoli-ly-maghreb', 'tobruk-ly-maghreb', 'annaba-dz-maghreb', 'msa-msa-msa', 'bordj_bou_arreridj-dz-maghreb', 'jijel-dz-maghreb', 'abu_dhabi-ae-gulf', 'fes-ma-maghreb', 'aleppo-sy-levant', 'suez-eg-nile_basin', 'ismailia-eg-nile_basin', 'samawah-iq-gulf', 'doha-qa-gulf', 'mansoura-eg-nile_basin', 'damascus-sy-levant', 'al_rayyan-qa-gulf', 'girga-eg-nile_basin', 'cairo-eg-nile_basin', 'buraidah-sa-gulf', 'riyadh-sa-gulf', 'karbala-iq-gulf', 'duhok-iq-gulf', 'el_arish-eg-nile_basin', 'oujda-ma-maghreb', 'aswan-eg-nile_basin', 'manama-bh-gulf', 'oran-dz-maghreb', 'jizan-sa-gulf', 'mahdia-tn-maghreb', 'jeddah-sa-gulf', 'agadir-ma-maghreb', 'beirut-lb-levant', 'tripoli-lb-levant', 'al_suwayda-sy-levant', 'tanta-eg-nile_basin', 'dammam-sa-gulf', 'mogadishu-so-gulf_aden', 'sfax-tn-maghreb', 'salalah-om-gulf', 'al_hudaydah-ye-gulf_aden', 'hurghada-eg-nile_basin', 'basra-iq-gulf', 'zagazig-eg-nile_basin', 'salt-jo-levant', 'rabat-ma-maghreb', 'sohar-om-gulf', 'abha-sa-gulf', 'fujairah-ae-gulf', 'mosul-iq-gulf', 'baghdad-iq-gulf', 'ariana-tn-maghreb', 'el_tor-eg-nile_basin', 'homs-sy-levant', 'beni_suef-eg-nile_basin', 'najran-sa-gulf', 'ramadi-iq-gulf', 'faiyum-eg-nile_basin', 'ouargla-dz-maghreb', 'ras_al_khaimah-ae-gulf', 'algiers-dz-maghreb', 'nouakchott-mr-maghreb', 'tabuk-sa-gulf', 'tunis-tn-maghreb', 'minya-eg-nile_basin', 'dhamar-ye-gulf_aden', 'sousse-tn-maghreb', 'erbil-iq-gulf', 'khasab-om-gulf', 'bouira-dz-maghreb', 'djibouti-dj-gulf_aden', 'ibb-ye-gulf_aden', 'al_madinah-sa-gulf', 'jerusalem-ps-levant', 'khenchela-dz-maghreb', 'qena-eg-nile_basin', 'jahra-kw-gulf', 'kairouan-tn-maghreb', 'damanhur-eg-nile_basin', 'alexandria-eg-nile_basin', 'port_said-eg-nile_basin', 'sanaa-ye-gulf_aden', 'dubai-ae-gulf', 'giza-eg-nile_basin', 'sulaymaniyah-iq-gulf', 'latakia-sy-levant', 'zarqa-jo-levant', 'sur-om-gulf', 'nizwa-om-gulf', 'aden-ye-gulf_aden', 'benghazi-ly-maghreb', 'bechar-dz-maghreb', 'gaza-ps-levant', 'misrata-ly-maghreb', 'tangier-ma-maghreb']),
                         'aggregated_city', )

AGGREGATED_LAYERS = [city_layer]


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
                 labels_aggregated=AGGREGATED_LABELS_EXTRA,
                 char_lm_dir=None,
                 word_lm_dir=None,
                 aggregated_layers=None,
                 aggregated_word_lm_dir=None,):
        if char_lm_dir is None:
            char_lm_dir = _CHAR_LM_DIR
        if word_lm_dir is None:
            word_lm_dir = _WORD_LM_DIR
        if aggregated_layers is None:
            aggregated_layers = _AGGREGATED_CHAR_LM_DIR

        # aggregated layer
        self._aggregated_layers = aggregated_layers

        self._labels = labels
        self._labels_extra = labels_extra
        self._labels_aggregated = labels_aggregated
        self._labels_sorted = sorted(labels)
        self._labels_extra_sorted = sorted(labels_extra)
        self._labels_aggregated_sorted = sorted(labels_aggregated)

        self._char_lms = collections.defaultdict(kenlm.Model)
        self._word_lms = collections.defaultdict(kenlm.Model)
        self._load_lms(char_lm_dir, word_lm_dir)

        self._aggregated_char_lms = collections.defaultdict(kenlm.Model)
        self._aggregated_word_lms = collections.defaultdict(kenlm.Model)
        self._load_aggregated_lms(
            aggregated_char_lm_dir, aggregated_word_lm_dir)

        self._is_trained = False

    def _load_aggregated_lms(self, char_lm_dir,  word_lm_dir):
        config = kenlm.Config()
        config.show_progress = False

        for label in self._labels_aggregated:
            char_lm_path = os.path.join(char_lm_dir, '{}.arpa'.format(label))
            word_lm_path = os.path.join(word_lm_dir, '{}.arpa'.format(label))
            self._aggregated_char_lms[label] = kenlm.Model(
                char_lm_path, config)
            self._aggregated_word_lms[label] = kenlm.Model(
                word_lm_path, config)

    def _load_lms(self, char_lm_dir, word_lm_dir):
        config = kenlm.Config()
        config.show_progress = False

        for label in self._labels:
            char_lm_path = os.path.join(char_lm_dir, '{}.arpa'.format(label))
            word_lm_path = os.path.join(word_lm_dir, '{}.arpa'.format(label))
            self._char_lms[label] = kenlm.Model(char_lm_path, config)
            self._word_lms[label] = kenlm.Model(word_lm_path, config)

    def _get_aggregated_char_lm_scores(self, txt):
        chars = _word_to_char(txt)
        return np.array([self._aggregated_char_lms[label].score(chars, bos=True, eos=True)
                         for label in self._labels_aggregated_sorted])

    def _get_aggregated_word_lm_scores(self, txt):
        return np.array([self._aggregated_word_lms[label].score(txt, bos=True, eos=True)
                         for label in self._labels_aggregated_sorted])

    def _get_aggregated_lm_feats(self, txt):
        word_lm_scores = self._get_aggregated_word_lm_scores(
            txt).reshape(1, -1)
        word_lm_scores = _normalize_lm_scores(word_lm_scores)
        char_lm_scores = self._get_aggregated_char_lm_scores(
            txt).reshape(1, -1)
        char_lm_scores = _normalize_lm_scores(char_lm_scores)
        feats = np.concatenate((word_lm_scores, char_lm_scores), axis=1)
        return feats

    def _get_aggregated_lm_feats_multi(self, sentences):
        feats_list = collections.deque()
        for sentence in sentences:
            feats_list.append(self._get_aggregated_lm_feats(sentence))
        feats_matrix = np.array(feats_list)
        feats_matrix = feats_matrix.reshape((-1, 52))
        return feats_matrix

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
        tokenized = [' '.join(simple_word_tokenize(dediac_ar(s)))
                     for s in sentences]
        sent_array = np.array(tokenized)
        x_trans = self._feat_union.transform(sent_array)
        x_trans_extra = self._feat_union_extra.transform(sent_array)
        x_trans_aggregated = self._feat_union_aggregated.transform(sent_array)
        x_predict_extra = self._classifier_extra.predict_proba(x_trans_extra)
        x_predict_aggregated = self._classifier_aggregated.predict_proba(
            x_trans_aggregated)
        x_lm_feats = self._get_lm_feats_multi(sentences)
        x_final = sp.sparse.hstack(
            (x_trans, x_lm_feats, x_predict_extra, x_predict_aggregated))
        return x_final

    def train(self, data_path=None,
              data_extra_path=None,
              data_aggregated_path=None,
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

        # Load training data and extract
        train_data = pd.read_csv(data_path, sep='\t', index_col=0)
        train_data_extra = pd.read_csv(data_extra_path, sep='\t', index_col=0)
        train_data_aggregated = pd.read_csv(
            data_aggregated_path, sep='\t', header=0)

        x = train_data['ar'].values
        y = train_data['dialect'].values
        x_extra = train_data_extra['ar'].values
        y_extra = train_data_extra['dialect'].values

        x_aggregated = train_data_aggregated['original_sentence'].values
        cols = ['dialect_city_id', 'dialect_country_id', 'dialect_region_id']
        train_data_aggregated['combined'] = train_data_aggregated[cols].apply(
            lambda row: '-'.join(row.values.astype(str)), axis=1)
        y_aggregated = train_data_aggregated['combined'].values

        # Build and train extra classifier
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

        # Build and train aggregated classifier
        self._label_encoder_aggregated = LabelEncoder()
        self._label_encoder_aggregated.fit(y_aggregated)
        y_trans = self._label_encoder_aggregated.transform(y_aggregated)

        word_vectorizer = TfidfVectorizer(lowercase=False,
                                          ngram_range=word_ngram_range,
                                          analyzer='word',
                                          tokenizer=lambda x: x.split(' '))
        char_vectorizer = TfidfVectorizer(lowercase=False,
                                          ngram_range=char_ngram_range,
                                          analyzer='char',
                                          tokenizer=lambda x: x.split(' '))
        self._feat_union_aggregated = FeatureUnion([('wordgrams', word_vectorizer),
                                                    ('chargrams', char_vectorizer)])
        x_trans = self._feat_union_aggregated.fit_transform(x_aggregated)

        self._classifier_aggregated = OneVsRestClassifier(MultinomialNB(),
                                                          n_jobs=n_jobs)
        self._classifier_aggregated.fit(x_trans, y_trans)

        # Build and train main classifier
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

    def eval(self, data_path=None, data_set='VALIDATION'):
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

        # Load eval data
        eval_data = pd.read_csv(data_path, sep='\t', index_col=0)
        x = eval_data['ar'].values
        y_true = eval_data['dialect'].values

        # Generate predictions
        x_prepared = self._prepare_sentences(x)
        y_pred = self._classifier.predict(x_prepared)
        print(y_pred)
        print(self._classifier.predict_proba(x_prepared))
        y_pred = self._label_encoder.inverse_transform(y_pred)

        # Get scores
        scores = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_micro': f1_score(y_true, y_pred, average='micro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'recall_micro': recall_score(y_true, y_pred, average='micro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'precision_micro': precision_score(y_true, y_pred,
                                               average='micro'),
            'precision_macro': precision_score(y_true, y_pred, average='macro')
        }

        return scores

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
