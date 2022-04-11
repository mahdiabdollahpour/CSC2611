# Mahdi Abdollahpour, 23/03/2022, 12:32 PM, PyCharm, project


def read_file(file_path):
    lines = []
    with open(file_path, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [line.replace('\n', '').split('\t') for line in lines]
    # print(lines)
    d = {line[0]: int(line[1]) for line in lines}
    return d
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


truth = read_file('./test_data_truth/task1/english.txt')


def explain(pred_path):
    pred = read_file(pred_path)
    # print(truth)
    # print(pred)
    tp = []
    fn = []
    fp = []
    tn = []
    for key in truth.keys():
        if truth[key] == 1 and pred[key] == 1:
            tp.append([key,freqs[key]])
        elif truth[key] == 1 and pred[key] == 0:
            fn.append([key,freqs[key]])
        elif truth[key] == 0 and pred[key] == 1:
            fp.append([key,freqs[key]])
        else:
            tn.append([key,freqs[key]])
    print(pred_path)
    print('tp',len(tp), tp)
    print('tn',len(tn), tn)
    print('fp', len(fp),fp)
    print('fn', len(fn),fn)
    print((len(tn) + len(tp)) / (len(fn) + len(fp) + len(tn) + len(tp)))
    return [tp,tn,fp,fn]
res1 = explain('./answers/ent_diff_2/answer/task1/english.txt')
res2 = explain('./answers/ents_diff/answer/task1/english.txt')

print('false positives that second one got right \n',[x for x in res1[2] if x in res2[1]])
print([x for x in res2[2] if x in res1[1]])


print('false negative that second one got right \n',[x for x in res1[3] if x in res2[0]])
print([x for x in res2[3] if x in res1[0]])
