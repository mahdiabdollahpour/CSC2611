import argparse
import zipfile
import re
import collections
import numpy as np
from six.moves import xrange
import random
import torch
import timeit
from torch.autograd import Variable
from models import SkipGramModel
from models import CBOWModel
from inference import save_embeddings
import time
from tqdm import tqdm
import pickle
import random


cmd_parser = argparse.ArgumentParser(description=None)
cmd_parser.add_argument('-p', '--plot', default='tsne.png',
                        help='Plotting output filename.')
cmd_parser.add_argument('-pn', '--plot_num', default=20, type=int,
                        help='Plotting data number.')
def get_targets(target_path):
    lines = []
    with open(target_path, mode='r') as f:
        lines = f.readlines()
    target_words = []

    for line in lines:
        word = line.replace('\n', '')
        target_words.append(word)
    return target_words

def get_top(word,embeddings,dictionary):
    scores = []
    # print(reverse_dictionary.keys())
    vector = embeddings[dictionary[word]]
    for key in dictionary.keys():
        score = vector.dot(embeddings[dictionary[key]])
        scores.append((key,score))
    scores = sorted(scores,key=lambda x: x[1],reverse=True)
    return scores

def tsne_plot(embeddings, num, reverse_dictionary, filename,dictionary):
    """Plot tSNE result of embeddings for a subset of words"""
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
    except ImportError as ex:
        print('Please install sklearn, matplotlib, and scipy to plot embeddings.')
        print(ex)
        return
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    # final_embeddings = embeddings.cpu()
    # targets = get_targets('./targets.txt')
    # top = []
    # for target in targets:
    #     tt = get_top(target,embeddings,dictionary)
    #     top.extend(tt[1:11])
    # print(top[1:11])
    # print(reverse_dictionary.keys())

    indexes = list(range(len(reverse_dictionary)))
    # random.shuffle(indexes)
    # indexes = [dictionary[t[0]] for t in top]
    
    low_dim_embs = tsne.fit_transform(embeddings[indexes[:num], :])
    low_dim_labels = [reverse_dictionary[i] for i in indexes[:num]]
    assert low_dim_embs.shape[0] >= len(low_dim_labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(low_dim_labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    print("Saving plot to:", filename)
    plt.savefig(filename)




if __name__ == '__main__':
    args = cmd_parser.parse_args()
    # dev = get_deivice(args.disable_cuda)
    # Data preprocessing
    # vocabulary = read_data(args.data)
    # print('Data size', len(vocabulary))
    # data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
    #                                                             args.size)
    # if not check_targets_included(args.target,dictionary):
    #     exit() 
    # vocabulary_size = min(args.size, len(count))
    # print('Vocabulary size', vocabulary_size)
    # word_count = [ c[1] for c in count]

    
    # Save result and plotting
    file = open('sw7/cbow_en_c2.bin', 'rb')
    data = pickle.load(file)
    final_embeddings = data['emb']
    dictionary = data['dict']
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    vocabulary_size = len(dictionary)
    norm = torch.sqrt(torch.cumsum(torch.mul(final_embeddings, final_embeddings), 1))
    nomalized_embeddings = (final_embeddings/norm).cpu().numpy()
    tsne_plot(embeddings=nomalized_embeddings,
              num=min(vocabulary_size, args.plot_num),
              reverse_dictionary=reverse_dictionary,
              filename=args.plot,dictionary=dictionary)
