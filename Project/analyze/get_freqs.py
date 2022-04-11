# Mahdi Abdollahpour, 25/03/2022, 05:53 PM, PyCharm, project


from word2vec import read_data
from models import CBOWModel
import torch
from scipy.stats import entropy
import pickle
from tqdm import tqdm
import numpy as np
import argparse
import json
import os


from word2vec import get_targets

if __name__ == '__main__':

    vocabulary1 = read_data("../starting_kit/test_data_public/english/corpus1/lemma/en_ccoha1.txt")
    vocabulary2 = read_data("../starting_kit/test_data_public/english/corpus2/lemma/en_ccoha2.txt")
    target_path = '../targets.txt'
    target_words = get_targets(target_path)
    result1 = {k:[] for k in target_words}
    result2 = {k:[] for k in target_words}
    result = {}
    for i, x in tqdm(enumerate(vocabulary1)):
        if x in target_words:
            result1[x].append(i)
    for i, x in tqdm(enumerate(vocabulary2)):
        if x in target_words:
            result2[x].append(i)

    for word in target_words:
        result[word]=[len(result1[word]), len(result2[word])]
    # for r in result:
    print(result)
