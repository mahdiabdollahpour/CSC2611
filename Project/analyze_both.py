import json
import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.stats.stats import pearsonr

parsa = {'bit': 0.0019245172851151127, 'thump': 0.0015863616313990248, 'stab': 0.0013399191836624968,
         'multitude': 0.0011623115336035772, 'tip': 0.0009499176287330879, 'record': 0.0009476029984204493,
         'quilt': 0.0009299023111563054, 'plane': 0.0008659853136054885, 'player': 0.000849457661244446,
         'stroke': 0.0008451868215904046, 'risk': 0.0007360014926814928, 'gas': 0.0007281766949160939,
         'fiction': 0.0006517776965624389, 'pin': 0.0006436477188291878, 'savage': 0.0006349649015306991,
         'prop': 0.0006037664021240063, 'twist': 0.0005870027442499426, 'rag': 0.0005831339829587279,
         'circle': 0.0005805912983706252, 'bag': 0.0005735102497116396, 'ounce': 0.0005504729079117698,
         'relationship': 0.0005067987179282474, 'attack': 0.0005031813831846144, 'part': 0.0004837346692315725,
         'word': 0.000439254287739832, 'ball': 0.00038388262706490206, 'donkey': 0.00034881072395043233,
         'edge': 0.00033459210008479623, 'head': 0.00031573224893299834, 'land': 0.0002742479821156163,
         'face': 0.00023878345596084483, 'tree': 0.00022472897975001072}


def read_json(json_file):
    # Opening JSON file
    with open(json_file) as file:
        data = json.load(file)
    data = data.replace("\'", "\"")
    data = json.loads(data)
    return data


def read_file(file_path, f=False):
    lines = []
    with open(file_path, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [line.replace('\n', '').split('\t') for line in lines]
    # print(lines)
    if f:
        d = {line[0]: float(line[1]) for line in lines}
    else:
        d = {line[0]: int(line[1]) for line in lines}
    return d


load_path = './output_files_rev'

truth = read_file('./test_data_truth/task1/english.txt')
truth_task2 = read_file('./test_data_truth/task2/english.txt', f=True)

# ents_diff = read_json(load_path + '/ents_diff.json')
# ents_diff_pos = read_json(load_path + '/ents_diff_pos.json')
# ents_diff_bin = read_json(load_path + '/ents_diff_bin.json')
# probs_diff = read_json(load_path + '/probs_diff.json')
# probs_diff_bin = read_json(load_path + '/probs_diff_bin.json')
# ents = read_json(load_path + '/ents.json')
# probs = read_json(load_path + '/probs.json')

freqs = {'attack_nn': [454, 833], 'bag_nn': [214, 899], 'ball_nn': [440, 878], 'bit_nn': [296, 622],
         'chairman_nn': [147, 683], 'circle_vb': [131, 245], 'contemplation_nn': [240, 111], 'donkey_nn': [118, 148],
         'edge_nn': [457, 1072], 'face_nn': [3394, 3932], 'fiction_nn': [202, 326], 'gas_nn': [155, 680],
         'graft_nn': [119, 109], 'head_nn': [3599, 4127], 'land_nn': [2321, 1624], 'lane_nn': [211, 289],
         'lass_nn': [111, 106], 'multitude_nn': [475, 131], 'ounce_nn': [208, 189], 'part_nn': [4410, 3213],
         'pin_vb': [114, 217], 'plane_nn': [278, 792], 'player_nn': [132, 939], 'prop_nn': [121, 147],
         'quilt_nn': [106, 189], 'rag_nn': [158, 208], 'record_nn': [420, 1188], 'relationship_nn': [130, 841],
         'risk_nn': [286, 643], 'savage_nn': [504, 133], 'stab_nn': [92, 117], 'stroke_vb': [110, 227],
         'thump_nn': [89, 127], 'tip_vb': [119, 241], 'tree_nn': [2322, 1596], 'twist_nn': [103, 186],
         'word_nn': [4387, 3166]}


def get_pairs(path1, path2, name, reverse=True, mode=3, log=False, plot=False):
    values = read_json(path1 + name)
    values2 = read_json(path2 + name)
    pairs = []
    data_plotx = []
    data_ploty = []
    data_ploty_gt = []
    names = []
    data_ploty_parsa = []
    for key in values.keys():
        # print(key,np.mean(probs[key]),'prob')
        if mode == 1:
            v = np.mean(values[key])
        elif mode == 2:
            v = np.mean(values2[key])
        elif mode == 4:
            v = max(np.mean(values[key]), np.mean(values2[key]))

        elif mode == 5:
            if freqs[key][1] > freqs[key][0]:
                v = np.mean(values[key])
            else:
                v = np.mean(values2[key])
        else:
            v = np.mean(values[key]) + np.mean(values2[key])
        # float("{:10.4f}".format(v))
        freq_change = ((freqs[key][1] - freqs[key][0]) / freqs[key][0])
        # pairs.append((key, v, truth[key],truth_task2[key], freqs[key], freqs[key][1] - freqs[key][0],
        #               freq_change))

        pairs.append((key,v, truth_task2[key]))

        # freq_change = max(freqs[key][1], freqs[key][0])
        # freq_change = str(freqs[key][1]) + ',' + str(freqs[key][0])
        names.append(key[:-3])
        data_plotx.append(freq_change)
        data_ploty.append(v)
        data_ploty_gt.append(truth_task2[key])
        # print( key[:-3])
        if key[:-3] in parsa.keys():

            data_ploty_parsa.append(parsa[key[:-3]] * 500)
        else:
            data_ploty_parsa.append(0)
    if plot:
        fig, ax = plt.subplots()

        ax.scatter(data_ploty_gt,data_ploty )
        # plt.title("title")
        plt.xlabel("GT LSC")
        plt.ylabel("Predicted LSC")
        for i, txt in enumerate(names):
            ax.annotate(txt, (data_ploty_gt[i], data_ploty[i]))
        plt.show()
        fig, ax = plt.subplots()
        ax.scatter(data_plotx, data_ploty)
        plt.xlabel("Freq. change %")
        plt.ylabel("Predicted LSC")
        # plt.scatter(data_plotx, data_ploty_gt)
        # plt.scatter(data_plotx, data_ploty_parsa)
        plt.show()
    pairs.sort(key=lambda y: y[1], reverse=reverse)

    avg = np.mean([p[1] for p in pairs])
    c = 0
    w = 0
    value = 1
    if reverse == False:
        value = -1
    preds = []
    for i, p in enumerate(pairs):
        preds.append(i)
    bin_predictions = {}
    for p in pairs:
        if (value) * (p[1] - avg) > 0:
            pred = 1

        else:
            pred = 0
        stat = 'correct'
        if pred == truth[p[0]]:
            c += 1
        else:
            stat = 'wrong'
            w += 1
        if log:
            print(p, pred,stat)
            # print(p[0][:-3],'&',"${}$".format('%.3f'%(p[1])),'&',"${}$".format(p[2]),'\\\\')
        bin_predictions[p[0]] = pred

    acc = c / (c + w)
    scores = [p[1] for p in pairs]
    task2_truth = [truth_task2[p[0]] for p in pairs]
    # print(scores)
    # print(task2_truth)
    # print(scores)
    # print(preds)
    # print(task2_truth)
    corr = pearsonr(scores,task2_truth)
    print(name, acc, c, w, avg, reverse,corr[0])
    # if acc >= 0.58:
    #     for pair in pairs:
    #         print(pair)
    print('-' * 20)
    return pairs, bin_predictions, scores


def create_files(name, pred1, pred2):
    targets = freqs.keys()
    path = './answers/' + name + '/answer'
    if not os.path.exists(path):
        os.makedirs(path + '/task1')
        os.makedirs(path + '/task2')
    with open(path + '/task1/english.txt', mode='w') as f:

        for word in targets:
            f.write(word + '\t' + str(pred1[word]) + '\n')

    with open(path + '/task2/english.txt', mode='w') as f:
        for word in targets:
            f.write(word + '\t' + str(pred2[word]) + '\n')


mode = 1
# ent_pairs, _, _ = get_pairs('./output_files', './output_files_rev', '/ents.json', mode=mode, log=False)
# ent_diff_pairs, pred1, pred2 = get_pairs('./output_files', './output_files_rev', '/ents_diff.json', mode=mode,
#                                          log=False, plot=True)
# ents_diff_pos_pairs, _, _ = get_pairs('./output_files', './output_files_rev', '/ents_diff_pos.json', mode=mode,
#                                       log=False)
# ents_diff_bin_pairs, _, _ = get_pairs('./output_files', './output_files_rev', '/ents_diff_bin.json', mode=mode)
# prob_pairs, _, _ = get_pairs('./output_files', './output_files_rev', '/probs.json', False, mode=mode)
# prob_diff_pairs, pd_pred1, pd_pred2 = get_pairs('./output_files', './output_files_rev', '/probs_diff.json', mode=mode,
#                                                 log=False)
prob_diff_bin_pairs, pdb_pred1, pdb_pred2 = get_pairs('./output_files', './output_files_rev', '/probs_diff_bin.json',
                                                      mode=mode, log=True,plot=False)
# prob_diff_abs, _, _ = get_pairs('./output_files', './output_files_rev', '/ents_diff_abs.json',
#                                                       mode=mode, log=True,plot=True)


# ens = {}
# for key in pd_pred2.keys():
#     ens[key] = pred1[key] *0.8 + pdb_pred1[key] *0.2
# avg = np.mean(list(ens.values()))
# print(avg,ens)
# c = 0
# w = 0
# for key in ens.keys():
#     if ens[key] - avg > 0:
#         pred = 1
#     else:
#         pred = 0
#     if pred==truth[key]:
#         c+=1
#     else:
#         w+=1
#
# print(c/(c+w))

# create_files('./probs_diff_bin_2', pdb_pred1, pdb_pred2)
