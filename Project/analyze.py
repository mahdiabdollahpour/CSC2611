import json
import matplotlib.pyplot as plt
import numpy as np
import os


def read_json(json_file):
    # Opening JSON file
    with open(json_file) as file:
        data = json.load(file)
    data = data.replace("\'", "\"")
    data = json.loads(data)
    return data


def read_file(file_path):
    lines = []
    with open(file_path, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [line.replace('\n', '').split('\t') for line in lines]
    # print(lines)
    d = {line[0]: int(line[1]) for line in lines}
    return d


truth = read_file('./test_data_truth/task1/english.txt')

load_path = './output_files_rev'
ents_diff = read_json(load_path + '/ents_diff.json')
ents_diff_pos = read_json(load_path + '/ents_diff_pos.json')
ents_diff_bin = read_json(load_path + '/ents_diff_bin.json')
probs_diff = read_json(load_path + '/probs_diff.json')
probs_diff_bin = read_json(load_path + '/probs_diff_bin.json')
# kls = read_json('./output_files/kls.json')
ents = read_json(load_path + '/ents.json')
probs = read_json(load_path + '/probs.json')

targets = list(probs.keys())
ent_pairs = []
ent_diff_pairs = []
ents_diff_pos_pairs = []
ents_diff_bin_pairs = []
prob_pairs = []
prob_diff_pairs = []
prob_diff_bin_pairs = []
# kl_pairs = []
for key in ents.keys():
    # print(key,np.mean(ents[key]))
    # x = lambda a : 1 if a>2 else 0
    # vv = [x(v) for v in ents[key]]
    ent_pairs.append((key, np.mean(ents[key]), truth[key]))
    # ent_pairs.append((key,np.mean(vv)))
ent_pairs.sort(key=lambda y: y[1], reverse=True)
for key in probs.keys():
    # print(key,np.mean(probs[key]),'prob')
    prob_pairs.append((key, np.mean(probs[key]), truth[key]))
prob_pairs.sort(key=lambda y: y[1])

for key in ents_diff.keys():
    # print(key,np.mean(probs[key]),'prob')
    ent_diff_pairs.append((key, np.mean(ents_diff[key]), truth[key]))
ent_diff_pairs.sort(key=lambda y: y[1], reverse=True)

for key in probs_diff.keys():
    # print(key,np.mean(probs[key]),'prob')
    prob_diff_pairs.append((key, np.mean(probs_diff[key]), truth[key]))
prob_diff_pairs.sort(key=lambda y: y[1])

for key in ents_diff_pos.keys():
    # print(key,np.mean(probs[key]),'prob')
    ents_diff_pos_pairs.append((key, np.mean(ents_diff_pos[key]), truth[key]))
ents_diff_pos_pairs.sort(key=lambda y: y[1])
for key in ents_diff_bin.keys():
    # print(key,np.mean(probs[key]),'prob')
    ents_diff_bin_pairs.append((key, np.mean(ents_diff_bin[key]), truth[key]))
ents_diff_bin_pairs.sort(key=lambda y: y[1], reverse=True)

for key in probs_diff_bin.keys():
    # print(key,np.mean(probs[key]),'prob')
    prob_diff_bin_pairs.append((key, np.mean(ents_diff_bin[key]), truth[key]))
prob_diff_bin_pairs.sort(key=lambda y: y[1], reverse=True)
# for key in kls.keys():
#     # print(key,np.mean(probs[key]),'prob')
#     kl_pairs.append((key, np.mean(kls[key]),truth[key]))
# kl_pairs.sort(key=lambda y: y[1])


print('ents')
for pair in ent_pairs:
    print(pair)
print('probs')
for pair in prob_pairs:
    print(pair)
# print('kl')
# for pair in kl_pairs:
#     print(pair)
print('prob_diff_pairs')
for pair in prob_diff_pairs:
    print(pair)
print('prob_diff_bin_pairs')
for pair in prob_diff_bin_pairs:
    print(pair)
print('ent_diff_pairs')
for pair in ent_diff_pairs:
    print(pair)
print('ent_diff_pos_pairs')
for pair in ents_diff_pos_pairs:
    print(pair)
print('ent_diff_bin_pairs')
for pair in ents_diff_bin_pairs:
    print(pair)


# print(ent_pairs)
# print(prob_pairs)
def plot_ents(words):
    for word in words:
        plt.figure()
        target_word = word
        plt.hist(ents[target_word], bins=30)
        plt.title(target_word)
        # plt.show() 

        # target_word= 'risk_nn'
        # plt.hist(ents[target_word],bins=30)
        # plt.title(target_word)
    plt.show()


def create_files(name, pairs, values):
    print(pairs)
    words = [p[0][:-3] for p in pairs]
    score = [p[1] for p in pairs]
    print(score)
    avg = np.mean(score)
    print(name, 'avg', avg)
    path = './answers/' + name + '/answer'
    if not os.path.exists(path):
        os.makedirs(path + '/task1')
        os.makedirs(path + '/task2')
    with open(path + '/task1/english.txt', mode='w') as f:

        for word in targets:
            if np.mean(values[word]) > avg:
                label = 1
            else:
                label = 0
            f.write(word + '\t' + str(label) + '\n')

    with open(path + '/task2/english.txt', mode='w') as f:
        for word in targets:
            label = np.mean(values[word])
            f.write(word + '\t' + str(label) + '\n')

    x_pos = np.arange(len(words))

    slope = 0
    intercept = avg
    trendline = intercept + (slope * x_pos)

    plt.plot(x_pos, trendline, color='red', linestyle='--')
    plt.bar(x_pos, score, align='center')
    plt.xticks(x_pos, words)
    plt.ylabel('Entropy')
    plt.show()


# create_files('prob', prob_pairs, probs)
# create_files('ent', ent_pairs, ents)
# create_files('probs_diff', prob_diff_pairs, probs_diff)
# create_files('probs_diff_bin', prob_diff_bin_pairs, probs_diff_bin)

# create_files('kl',kls, kl_pairs)
# create_files('ents_diff', ent_diff_pairs, ents_diff)
# create_files('ents_diff_bin', ents_diff_bin_pairs, ents_diff_bin)
# create_files('ents_diff_pos', ents_diff_pos_pairs, ents_diff_pos)

# plot_ents(['stroke_vb','chairman_nn','edge_nn','head_nn'])

# words = [p[0][:-3] for p in prob_pairs]
# score = [p[1] for p in prob_pairs]
# avg = np.mean(score)


# calculate slope and intercept for the linear trend line
# slope, intercept = np.polyfit(x_pos, score, 1)
