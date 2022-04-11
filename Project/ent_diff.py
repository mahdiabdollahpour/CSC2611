# Mahdi Abdollahpour, 18/03/2022, 11:00 PM, PyCharm, project

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
cmd_parser.add_argument('-o1', '--output1', default='embeddings.bin',
                        help='Output embeddings filename.')
cmd_parser.add_argument('-o2', '--output2', default='embeddings.bin',
                        help='Output embeddings filename.')
cmd_parser.add_argument('-sp', '--save_path', default='./output_files',
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


def get_model(args, output):
    vocabulary_size = args.size
    embedding_dim = args.embedding_dim
    device = 'cpu'
    model = CBOWModel(device, vocabulary_size, embedding_dim)
    head_tail = os.path.split(output)
    model.load_state_dict(torch.load(os.path.join(head_tail[0], 'model_' + head_tail[1])))
    # print(model.keys())
    model.eval()
    return model


def get_context(w, i, vocab,dict):
    context = []

    # print(dictionary.keys())
    for j in range(i - w, i):
        if vocab[j] in dict.keys():
            context.append(dict[vocab[j]])

        else:
            context.append(dict['UNK'])
    for j in range(i + 1, i + w + 1):
        if vocab[j] in dict.keys():
            context.append(dict[vocab[j]])
        else:
            context.append(dict['UNK'])
    context_tensor = torch.tensor([context])
    return context_tensor


if __name__ == '__main__':
    args = cmd_parser.parse_args()
    target_path = args.target
    vocabulary = read_data(args.data)
    # vocabulary = read_data("./starting_kit/test_data_public/english/corpus1/lemma/en_ccoha1.txt")
    file = open(args.output1, 'rb')
    file2 = open(args.output2, 'rb')
    # 'cbow_en_c2.bin'
    data = pickle.load(file)
    data2 = pickle.load(file2)
    # final_embeddings = data['emb']
    dictionary = data['dict']
    dictionary2 = data2['dict']
    # if dictionary == dictionary2:
    #     print('dictionaries match')
    # else:
    #     print('dictionaries dont match')
    # print(dictionary['ungovern'])
    # print(dictionary2['ungovern'])
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    reversed_dictionary2 = dict(zip(dictionary2.values(), dictionary2.keys()))
    model1 = get_model(args, args.output1)
    model2 = get_model(args, args.output2)
    target_words = get_targets(target_path)
    w = args.skip_window
    # print(logit.shape)
    ents = {key: [] for key in target_words}
    ents_diff = {key: [] for key in target_words}
    probs = {key: [] for key in target_words}
    probs_diff = {key: [] for key in target_words}
    probs_diff_bin = {key: [] for key in target_words}
    kls = {key: [] for key in target_words}
    ents_diff_pos = {key: [] for key in target_words}
    ents_diff_abs = {key: [] for key in target_words}
    ents_diff_bin = {key: [] for key in target_words}
    for i in tqdm(range(w - 1, len(vocabulary) - w)):
        if vocabulary[i] in target_words:
            target_word = vocabulary[i]
            # print(context_tensor.shape)
            # index = reversed_dictionary[vocabulary_size[i]]
            context_tensor = get_context(w, i, vocabulary, dictionary)
            context_tensor2 = get_context(w, i, vocabulary, dictionary2)
            with torch.no_grad():
                logit1, prob1 = model1(context_tensor, get_prob=True)
                logit2, prob2 = model2(context_tensor2, get_prob=True)
                # print(logit.shape)
                prob1 = prob1.numpy()
                prob2 = prob2.numpy()
                ent1 = entropy(prob1[0, :])
                ent2 = entropy(prob2[0, :])
                p1 = prob1[0, dictionary[target_word]]
                p2 = prob2[0, dictionary2[target_word]]
                # kl = entropy(prob1[0, :],prob2[0, :])
                if ent2 - ent1 > 0:
                    ents_diff_pos[target_word].append(ent2 - ent1)
                    ents_diff_bin[target_word].append(1)
                else:
                    ents_diff_bin[target_word].append(0)
                ents_diff[target_word].append(ent2-ent1)
                ents_diff_abs[target_word].append(np.abs(ent2-ent1))

                # print(prob1)
                if p2 < p1:

                    probs_diff_bin[target_word].append(1)
                else:
                    probs_diff_bin[target_word].append(0)
                probs_diff[target_word].append(p1-p2)


                ents[target_word].append(ent2)
                probs[target_word].append(p2)
                # kls[vocabulary[i]].append(kl)
    # with open('probs.json', 'w') as fp:
    #     json.dump(str(probs), fp)


    save_path = args.save_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    with open(save_path+'/ents_diff.json', 'w') as fp:
        json.dump(str(ents_diff), fp)
    with open(save_path+'/ents.json', 'w') as fp:
        json.dump(str(ents), fp)
    with open(save_path+'/probs.json', 'w') as fp:
        json.dump(str(probs), fp)
    with open(save_path+'/probs_diff.json', 'w') as fp:
        json.dump(str(probs_diff), fp)
    with open(save_path+'/ents_diff_pos.json', 'w') as fp:
        json.dump(str(ents_diff_pos), fp)
    with open(save_path+'/ents_diff_abs.json', 'w') as fp:
        json.dump(str(ents_diff_abs), fp)
    with open(save_path+'/ents_diff_bin.json', 'w') as fp:
        json.dump(str(ents_diff_bin), fp)
    with open(save_path+'/probs_diff_bin.json', 'w') as fp:
        json.dump(str(probs_diff_bin), fp)
    with open(save_path+'/kls.json', 'w') as fp:
        json.dump(str(kls), fp)

    ent_pairs = []
    prob_pairs = []
    kl_pairs = []
    # for key in ents.keys():
    #     # print(key,np.mean(ents[key]))
    #     ent_pairs.append((key, np.mean(ents[key])))
    # ent_pairs.sort(key=lambda y: y[1], reverse=True)
    # for key in probs.keys():
    #     # print(key,np.mean(probs[key]),'prob')
    #     prob_pairs.append((key, np.mean(probs[key])))
    # prob_pairs.sort(key=lambda y: y[1])

    # for key in kls.keys():
    #     # print(key,np.mean(probs[key]),'prob')
    #     kl_pairs.append((key, np.mean(kls[key])))
    #     kl_pairs.sort(key=lambda y: y[1])
    # print(ent_pairs)
    # print(prob_pairs)
    # print(kl_pairs)
