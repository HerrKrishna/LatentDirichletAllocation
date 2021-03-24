import sys
import os
import yaml
import numpy as np
from scipy.special import psi

def expectation_of_log(input):
    """
    Implements the equations [6] from Hoffman et. al. 2010
    """
    if (len(input.shape) == 1):
        return(psi(input) - psi(np.sum(input)))

    return (psi(input) - psi(np.sum(input, 1))[:, np.newaxis])

class LDA:

    def __init__(self, vocab, nr_of_docs, config):
        self._vocab = vocab
        self._voc_size = len(vocab)
        self._nr_of_docs = nr_of_docs
        self._tau = config['tau0']
        self._kappa = config['kappa']
        self._nr_of_topics = config['nr_of_topics']

        # doc_topic_prior is called alpha in Hoffmann et al. 2010
        if 'doc_topic_prior' in config.keys():
            self._doc_topic_prior = config['doc_topic_prior']
        else:
            self._doc_topic_prior = 1 / self._nr_of_topics

        # topic_word_prior is called eta in Hoffmann et al. 2010
        if 'topic_word_prior' in config.keys():
            self._topic_word_prior = config['topic_word_prior']
        else:
            self._topic_word_prior = 1 / self._nr_of_topics

        self.threshold = config['threshold']
        self._steps = 1

        # this is a nr_of_topics x voc_size Matrix that contains the topic vectors
        # Hoffman et. al. 2010 refer to this as lambda
        self._topics = np.random.gamma(100., 1. / 100., (self._nr_of_topics, self._voc_size))

        #_expectation_beta is E(log(beta))
        self._expectation_beta = expectation_of_log(self._topics)
        #_exp_beta is exp(E(log(beta)))
        self._exp_beta = np.exp(self._expectation_beta)


    def get_word_counts_and_ids(self, doc):

        id_count_dict = {}
        for word in doc:
            try:
                word_id = self._vocab.index(word)
            except ValueError:
                word_id = 0

            if not word_id in id_count_dict:
                id_count_dict[word_id] = 0
            id_count_dict[word_id] += 1
        return np.array(list(id_count_dict.keys())), np.array(list(id_count_dict.values()))

    def get_document_topics(self, docs):

        doc_topics, _ = self.e_step(docs)
        # this is for normalization
        sums = np.sum(doc_topics, -1)
        doc_topics = doc_topics / sums[:, None]
        return doc_topics


    def e_step(self, batch):

        remember_for_update = np.zeros(self._topics.shape)

        D = len(batch)
        # this is a DxK matrix
        # D documents, each of which is a prob distribution over K topics
        # in Hoffman et. al. 2013 this is refered to as gamma
        # TODO: find out why this gamma
        doc_topic_matrix = np.random.gamma(100., 1. / 100., (D, self._nr_of_topics))

        expectation_theta = expectation_of_log(doc_topic_matrix)
        exp_theta = np.exp(expectation_theta)
        exp_beta = self._exp_beta

        # we iterate over the documents in the batch
        # for each document we update its vector in the
        # doc_topic matrix based on the word counts in
        # the document
        for i, doc in enumerate(batch):
            word_ids, word_counts = self.get_word_counts_and_ids(doc)
            document_vector = doc_topic_matrix[i, :]
            # exp_beta is a nr_of_topics x nr_of_words_in_current_doc matrix
            exp_beta_this_doc = exp_beta[:, word_ids]
            #exp_theta is a vector of length nr_of_topics
            exp_theta_this_doc = exp_theta[i, :]
            converged = False
            norm_phi = np.dot(exp_theta_this_doc, exp_beta_this_doc) + 1e-100
            for j in range(100):
                if converged:
                    break

                remember_document_vector = document_vector
                # these are the updates in line 7 and 8 in Hoffmann et. al. 2010
                #
                # np.dot(word_counts / norm_phi, exp_beta_this_doc.T)
                # =
                # sum_w(word_counts_dw / norm_phi * exp( E(log(beta_dwk)))
                #
                # exp_theta_this_doc * np.dot(word_counts / norm_phi, exp_beta_this_doc.T)
                # = exp(E(log(theta_dk)) * sum_w( word_counts_dw / norm_phi * exp( E(log(beta_dwk)))
                # = sum_w(word_counts_dw / norm_phi * exp(E(log(theta_dk)) * exp( E(log(beta_dwk)))
                # = sum_w(word_counts_dw * exp(E(log(theta_dk)) + E(log(beta_dk)) / norm_phi)
                # = sum_w(normalized_word_counts_dw * phi_dw )
                # norm_phi is a normalizer. We need that since phi is proportional, not equal to
                # exp(E(log(theta_dk)) + E(log(beta_dk))

                document_vector = self._doc_topic_prior + (exp_theta_this_doc *
                                                           np.dot(word_counts / norm_phi, exp_beta_this_doc.T))

                exp_theta_this_doc = np.exp(expectation_of_log(document_vector))
                norm_phi = np.dot(exp_theta_this_doc, exp_beta_this_doc) + 1e-100
                avg_change = np.mean(abs(document_vector - remember_document_vector))
                converged = avg_change < self.threshold

            doc_topic_matrix[i, :] = document_vector
            # Contribution of document d to the expected sufficient
            # statistics for the M step.
            remember_for_update[:, word_ids] += np.outer(exp_theta_this_doc, word_counts / norm_phi)


        # remember_for_update[k, w] = \sum_d n_{dw} * phi_{dwk}
        # = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.
        remember_for_update = remember_for_update * self._exp_beta

        return doc_topic_matrix, remember_for_update

    def get_topic_vectors(self):
        return self._topics

    def fit_batch(self, batch):

        learning_rate = pow(self._tau + self._steps, -self._kappa)

        # update document-topic matrix for this batch
        doc_topic_matrix, sstats = self.e_step(batch)

        # Update lambda based on this batch
        # this is the update described in equation 8
        # in Hoffmann et. al. 2010
        ratio = self._nr_of_docs / len(batch)
        lambda_update = self._topic_word_prior + ratio * sstats
        self._topics = self._topics * (1 - learning_rate) + learning_rate * lambda_update
        self._expectation_beta = expectation_of_log(self._topics)
        self._exp_beta = np.exp(self._expectation_beta)
        self._steps += 1






