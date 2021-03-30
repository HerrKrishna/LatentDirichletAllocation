import sys
import os
import yaml
import itertools
import numpy as np
from data_reading import get_vocab, get_reddit, get_reuters
from model import LDA
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

def train(model, corpus, n_epochs, batch_size):

    count = 1
    for epoch in range(n_epochs):
        print(f'Epoch {epoch}:')
        for i in tqdm(range(0, len(corpus) - batch_size, batch_size)):
            batch = corpus[i: i + batch_size]
            model.fit_batch(batch=batch)
            count += 1

def get_clf_score(train_vectors, test_vectors, train_labels, test_labels):

    clf = RandomForestClassifier(max_depth=20, random_state=0)
    clf.fit(train_vectors, train_labels)

    pred = clf.predict(test_vectors)

    score = accuracy_score(test_labels, pred)
    return score

def get_k_means_score(vectors, train_labels):

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
    train_split, val_split, test_split = get_reddit(config['dataset'])
    train_corpus, train_subreddits, train_labels = train_split
    val_corpus, val_subreddits, val_labels = val_split
    test_corpus, test_subreddits, test_labels = test_split

    variable_params = config['variable_params']
    param_names = list(variable_params.keys())
    param_values = list(variable_params.values())
    del config['variable_params']

    hyper_params = []
    for i, name in enumerate(param_names):
        parameter_list = []
        for value in param_values[i]:
            parameter_list.append((name, value))
        hyper_params.append(parameter_list)

    hyper_param_combinations = list(itertools.product(*hyper_params))
 
    max_score = 0
    for combination in hyper_param_combinations:
        for item in combination:
            config[item[0]] = item[1]
	
        #hyper_param_dict = {**hyper_param_dict, **config}
        print(config)
    
        lda = LDA(vocab, len(train_corpus), config)
        # fit the model to the corpus
        train(lda, train_corpus, config['n_epochs'], config['batch_size'])
        # get lda-vectors
        lda_train = lda.get_document_topics(train_corpus)
        lda_val = lda.get_document_topics(val_corpus)

        lda_result = get_clf_score(lda_train, lda_val , train_labels, val_labels)
        print(lda_result)
        if lda_result > max_score:
            best_lda = lda
            best_config = config.copy()
            max_score = lda_result
    

    # get lda-vectors
    lda_train = best_lda.get_document_topics(train_corpus)
    lda_test = best_lda.get_document_topics(test_corpus)
    
    lda_result = get_clf_score(lda_train, lda_test , train_labels, test_labels)

    # get tf-idf-vectors
    vectorizer = TfidfVectorizer()
    tf_idf_train = vectorizer.fit_transform([' '.join(doc) for doc in train_corpus])
    tf_idf_test = vectorizer.transform(' '.join(doc) for doc in test_corpus)

    tf_idf_result = get_clf_score(tf_idf_train, tf_idf_test, train_labels, test_labels)
    print(best_config)
    print('LDA: ', lda_result)
    print('TF-IDF: ', tf_idf_result)
    return lda_result, tf_idf_result

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

    reddit_exp(config)
    #reuters_exp(config)
