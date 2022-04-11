## CSC2611 Project and Lab Assignments



# Project Abstract
Linguistic studies show that word meanings
are in constant change through time and de-
tecting the semantic change is essential for ma-
chine comprehension. Recent works on Lex-
ical Semantic Change (LSC) detection focus
on measuring the distance between word em-
bedding through time or modeling the senses
of the words through time. To the best of our
knowledge none of the previous works has fo-
cused on context uncertainty for LSC. In this
project, we explore different uncertainty mea-
sures to identify LSC. Given two diachronic
corpora, we train word embeddings for each of
them. Then given context words in one corpus,
we assume the prediction uncertainty under the
model trained on the other corpus, would be
higher for words that has not been with this
corpus. We evaluate our method on SemEval
2020 workshop benchmark and analyze the re-
sults. We found that frequency increase of the
word can be detected as semantic change in
our method. Our best method 56.75 accuracy
on subtask 1 and 0.251 correlation on subtask
2 of the SemEval benchmark.



### Download the data
download the data and put the starting_kit in the project directory

### Train embeddings

```
sh Project\experiments\train_en1.sh
sh Project\experiments\train_en2.sh
```

### Compute LSC measures
```
sh Project\experiments\cp_diff_en.sh
```

### Print the result
```
sh Project\analyze_both.py
```

Word2Vec training code is from: https://github.com/ksang/word2vec
