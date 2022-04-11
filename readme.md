## CSC2611 Project and Lab Assignments


# Running the project

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
