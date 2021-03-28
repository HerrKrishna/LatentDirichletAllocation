import os
import random
import numpy as np

def get_vocab(directory, vocab_size):

    vocab = []
    with open(os.path.join(directory, 'vocabulary'), 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines[:vocab_size]:
        vocab.append(line.replace('\n', ''))

    return vocab


def get_reuters(split):
    docs = []
    labels = []
    filenames = os.listdir('data/reuters/' + split)
    for filename in filenames:
        with open('data/reuters/' + split + '/' + filename, 'r', encoding='utf-8') as f:
            line = f.readline()
            doc = line.split('\t')[0].strip(' ').split(' ')
            labels_this_doc = line.split('\t')[1].strip(' ').split(' ')
            labels_this_doc = [int(label) for label in labels_this_doc]
            docs.append(doc)
            labels.append(labels_this_doc)

    # shuffle the dataset
    conc = list(zip(docs, labels))
    random.shuffle(conc)
    docs, labels = zip(*conc)
    docs = list(docs)
    labels = list(labels)

    lab = np.zeros([len(labels),90], dtype=np.int)
    print(lab)

    for i, set in enumerate(labels):
        lab[i, set] = 1
    print(lab[10:,:])

    return docs, lab

def get_corpus(dataset, split):

    with open('data/' + dataset + '/' + split, 'r') as f:
        lines = f.readlines()
    docs = []
    labels = []
    for line in lines:
        line = line.strip('\n')
        docs.append(line.split('\t')[0].split(' '))
        labels.append(int(line.split('\t')[1]))

    return docs, labels



def get_reddit(dataset, namelist = [], n_labels=None):

    dir_name = os.path.join('data/', dataset)
    docs = []
    subreddits = []
    seen_subreddits = []
    labels = []
    count = 0

    for filename in os.listdir(dir_name):
        count += 1

        if n_labels is not None:
            if count > n_labels:
                break

        if (len(namelist) > 0 and filename not in namelist) or filename.endswith('vocabulary'):
            continue

        with open(os.path.join(dir_name, filename), 'r', encoding='utf-8') as f:
            lines = f.readlines()

        subreddit = filename
        for line in lines:
            line = line.replace('\n', '')
            doc = line.split(' ')
            docs.append(doc)
            subreddits.append(subreddit)
            if not subreddit in seen_subreddits:
                seen_subreddits.append(filename)

            labels.append(seen_subreddits.index(filename))

    # shuffle the dataset
    conc = list(zip(docs, subreddits, labels))
    random.shuffle(conc)
    docs, subreddits, labels = zip(*conc)
    docs = list(docs)
    subreddits = list(subreddits)
    labels = list(labels)

    full_size = len(docs)
    test_size = int(full_size / 10)

    return (docs[:8*test_size], subreddits[:8*test_size], labels[:8*test_size]),\
           (docs[8*test_size:9*test_size], subreddits[8*test_size:9*test_size], labels[8*test_size:9*test_size]),\
           (docs[9*test_size:], subreddits[9*test_size:], labels[9*test_size:]),
