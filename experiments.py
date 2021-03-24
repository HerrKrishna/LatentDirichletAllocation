import sys
import os
import yaml
import numpy as np
from data_reading import get_vocab, get_reddit, get_reuters
from model import LDA
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train(model, corpus, batch_size):

    count = 1
    for i in range(0, len(corpus) - batch_size, batch_size):
        print(f'Batch {count} of {int(len(corpus) / batch_size)}')
        batch = corpus[i: i + batch_size]
        model.fit_batch(batch=batch)
        count += 1

def get_clf_score(lda, train_corpus, test_corpus, train_labels, test_labels):
    train_doc_topics = lda.get_document_topics(train_corpus)

    clf = RandomForestClassifier(max_depth=20, random_state=0)
    clf.fit(train_doc_topics, train_labels)

    test_doc_topics = lda.get_document_topics(test_corpus)
    pred = clf.predict(test_doc_topics)

    score = accuracy_score(test_labels, pred)
    return score

def get_k_means_score(lda, train_corpus, train_labels):

    train_doc_topics = lda.get_document_topics(train_corpus)
    score = 0
    for i in range(100):
        kmeans = KMeans(n_clusters=18, random_state=0).fit(train_doc_topics)

        true_labels = np.array(train_labels)
        labels_pred = kmeans.labels_

        score += adjusted_rand_score(true_labels, labels_pred)

    print(score / 100)


def reddit_exp(config):

    vocab = get_vocab(os.path.join('data', config['dataset']), config['vocab_size'])
    train_split, test_split = get_reddit(config['dataset'], split=9 / 10)
    train_corpus, train_subreddits, train_labels = train_split
    test_corpus, test_subreddits, test_labels = test_split

    lda = LDA(vocab, len(train_corpus), config)
    # fit the model to the corpus
    train(lda, train_corpus, config['batch_size'])


    print(get_clf_score(lda, train_corpus, test_corpus, train_labels, test_labels))
    #print(get_k_means_score(lda, train_corpus, train_labels))

def reuters_exp(config):

    vocab = get_vocab(os.path.join('data', config['dataset']), config['vocab_size'])
    train_corpus, train_labels = get_reuters('training')
    print(len(train_corpus))
    test_corpus, test_labels = get_reuters('test')

    print(len(train_corpus))

    lda = LDA(vocab, len(train_corpus), config)
    # fit the model to the corpus
    train(lda, train_corpus, config['batch_size'])
    print(get_clf_score(lda, train_corpus, test_corpus, train_labels, test_labels))
    print(get_k_means_score(lda, train_corpus, train_labels))

if __name__ == '__main__':

    config_path = sys.argv[1]
    with open(config_path) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    #reddit_exp(config)
    reuters_exp(config)
