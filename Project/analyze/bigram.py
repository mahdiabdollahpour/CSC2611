# Mahdi Abdollahpour, 25/03/2022, 07:09 PM, PyCharm, project

import nltk
from nltk.corpus import brown
import numpy as np
from nltk.corpus import brown, stopwords
import string
from tqdm import tqdm
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
from scipy.sparse import coo_matrix, lil_matrix
from collections import Counter
from word2vec import read_data
# puncs = [p for p in string.punctuation]
# nltk.download('stopwords')
# stopwords = stopwords.words('english')

# nltk.download('brown')
# bw = brown.words()
# print(bw)
from nltk.util import ngrams
from word2vec import get_targets

vocabulary1 = read_data("../starting_kit/test_data_public/english/corpus1/lemma/en_ccoha1.txt")
vocabulary2 = read_data("../starting_kit/test_data_public/english/corpus2/lemma/en_ccoha2.txt")

word_set = list(set(vocabulary1 + vocabulary2))
word2idx = {}
for i, w in enumerate(word_set):
    word2idx[w] = i

target_path = '../targets.txt'
target_words = get_targets(target_path)
target2idx = {}
for i, w in enumerate(target_words):
    target2idx[w] = i


def compute_bigram(text):
    # matrix = np.zeros((len(word_set), len(word_set)))
    bigrams_counts = Counter(ngrams(text, 2))
    # bigrams_counts = Counter(ngrams(bw, 2))

    # M1 = lil_matrix((len(word_set), len(word_set)))
    M1 = np.zeros((len(target_words), len(word_set)))
    for bigram in tqdm(bigrams_counts.keys()):
        if bigram[1].lower() in target_words and bigram[0].lower() in word_set:
            M1[target2idx[bigram[0].lower()], word2idx[bigram[1].lower()]] = bigrams_counts[bigram]
    return M1


m1 = compute_bigram(" ".join(vocabulary1))
m2 = compute_bigram(" ".join(vocabulary2))

print(np.sum(np.abs(m1 - m2),axis=0))
