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

cmd_parser = argparse.ArgumentParser(description=None)
# Data arguments
cmd_parser.add_argument('-d', '--data', default='data/text8.zip',
                        help='Data file for word2vec training.')
cmd_parser.add_argument('-t', '--target', default='targets.txt',
                        help='target words filename.')
cmd_parser.add_argument('-o', '--output', default='embeddings.bin',
                        help='Output embeddings filename.')

cmd_parser.add_argument('-s', '--size', default=50000, type=int,
                        help='Vocabulary size.')
# Model training arguments
cmd_parser.add_argument('-sw', '--skip_window', default=1, type=int,
                        help='How many words to consider left and right.')
cmd_parser.add_argument('-ed', '--embedding_dim', default=128, type=int,
                        help='Dimension of the embedding vector.')


def get_targets(target_path):
    lines = []
    with open(target_path, mode='r') as f:
        lines = f.readlines()
    target_words = []

    for line in lines:
        word = line.replace('\n', '')
        target_words.append(word)
    return target_words


if __name__ == '__main__':
    args = cmd_parser.parse_args()
    target_path = args.target
    vocabulary = read_data(args.data)
    # vocabulary = read_data("./starting_kit/test_data_public/english/corpus1/lemma/en_ccoha1.txt")
    file = open(args.output, 'rb')
    # 'cbow_en_c2.bin'
    data = pickle.load(file)
    final_embeddings = data['emb']
    dictionary = data['dict']
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    vocabulary_size = args.size
    embedding_dim = args.embedding_dim
    device = 'cpu'
    model = CBOWModel(device, vocabulary_size, embedding_dim)
    head_tail = os.path.split(args.output)
    model.load_state_dict(torch.load(os.path.join(head_tail[0], 'model_' + head_tail[1])))
    # print(model.keys())
    model.eval()
    target_words = get_targets(target_path)
    w = args.skip_window
    # print(logit.shape)
    ents = {key: [] for key in target_words}
    probs = {key: [] for key in target_words}
    for i in tqdm(range(w - 1, len(vocabulary) - w)):
        if vocabulary[i] in target_words:
            context = []
            # print(dictionary.keys())
            for j in range(i - w, i):
                if vocabulary[j] in dictionary.keys():
                    context.append(dictionary[vocabulary[j]])
                else:
                    context.append(dictionary['UNK'])
            for j in range(i + 1, i + w + 1):
                if vocabulary[j] in dictionary.keys():
                    context.append(dictionary[vocabulary[j]])
                else:
                    context.append(dictionary['UNK'])
            context_tensor = torch.tensor([context])
            # print(context_tensor.shape)
            # index = reversed_dictionary[vocabulary_size[i]]
            with torch.no_grad():
                logit, prob = model(context_tensor, get_prob=True)
                # print(logit.shape)
                prob = prob.numpy()
                ent = entropy(prob[0, :])
                ents[vocabulary[i]].append(ent)
                probs[vocabulary[i]].append(prob[0, dictionary[vocabulary[i]]])
                # if ent >= 1.52 and vocabulary[i]=='circle_vb':
                #     print()
                #     print('tp',ent,vocabulary[i-7:i+7])
                # if ent <= 1.52 and vocabulary[i] == 'circle_vb':
                #     print()
                #     print('fp',ent, vocabulary[i - 7:i + 7])
    with open('probs.json', 'w') as fp:
        json.dump(str(probs), fp)
    with open('ents.json', 'w') as fp:
        json.dump(str(ents), fp)

    ent_pairs = []
    prob_pairs = []
    for key in ents.keys():
        # print(key,np.mean(ents[key]))
        ent_pairs.append((key, np.mean(ents[key])))
    ent_pairs.sort(key=lambda y: y[1], reverse=True)
    for key in probs.keys():
        # print(key,np.mean(probs[key]),'prob')
        prob_pairs.append((key, np.mean(probs[key])))
    prob_pairs.sort(key=lambda y: y[1])
    print(ent_pairs)
    print(prob_pairs)
