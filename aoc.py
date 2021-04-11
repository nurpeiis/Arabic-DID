import csv
import pandas as pd
import os
from camel_tools.tokenizers import word as tokenizer
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.append('/Users/nurpeiis/Desktop/Capstone/hierarchical-did/utils')
from data_process import DataProcess
import xml.etree.ElementTree as et

# Provide the information about the source from which the data will be processed.
dp = DataProcess('../data_processed_splited/aoc/', 'mechanical_turk_annotators', 'news_comments', 'https://www.cis.upenn.edu/~ccb/data/AOC-dialectal-annotations.zip', 'aoc', {},{},0, 'corpus', 'manual')


dp.save_file('dialect_alghad-segs.tsv', dp.preprocess('../../data_raw/AOC-dialectal-annotations/dialect_alghad-segs.txt.norm', '', '', '', 'levant', header=None))
dp.save_file('dialect_youm7-segs.tsv', dp.preprocess('../../data_raw/AOC-dialectal-annotations/dialect_youm7-segs.txt.norm', '', '', 'eg', 'nile_basin', header=None))
dp.save_file('dialect_alriyadh-segs.tsv', dp.preprocess('../../data_raw/AOC-dialectal-annotations/dialect_alriyadh-segs.txt.norm', '', '', '', 'gulf', header=None))
dp.save_file('MSA_alghad-segs.tsv', dp.preprocess('../../data_raw/AOC-dialectal-annotations/MSA_alghad-segs.txt.norm', 'msa', 'msa', 'msa', 'msa', header=None))
dp.save_file('MSA_youm7-segs.tsv', dp.preprocess('../../data_raw/AOC-dialectal-annotations/MSA_youm7-segs.txt.norm', 'msa', 'msa', 'msa', 'msa', header=None))
dp.save_file('MSA_alriyadh-segs.tsv', dp.preprocess('../../data_raw/AOC-dialectal-annotations/MSA_alriyadh-segs.txt.norm', 'msa', 'msa', 'msa', 'msa', header=None))


files = ['dialect_alghad-segs.tsv', 'dialect_youm7-segs.tsv', 'dialect_alriyadh-segs.tsv', 'MSA_alghad-segs.tsv', 'MSA_youm7-segs.tsv', 'MSA_alriyadh-segs.tsv']
for file in files:
    df_train, df_dev, df_test = dp.split(file, 0.8, 0.1, 0.1)
    dp.save_file('train_'+file, df_train)
    dp.save_file('dev_'+file, df_dev)
    dp.save_file('test_'+file, df_test)

dp.save_features('../datasets_splited_features.tsv')


from lxml import etree

parser = etree.XMLParser(recover=True)

file = 'AOC_alghad-sample_articles.xml'
xtree = etree.parse('../../data_raw/AOC/AOC_v1.1/{}'.format(file))
xroot = xtree.getroot()
print(xroot)
rows = []
for node in xroot.findall("seg"):
    l = node.text
    rows.append({"original_sentence": line})
    print(l)


