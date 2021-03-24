from convokit import Corpus, download
import sys
import os
import spacy
import nltk
from nltk.corpus import brown, reuters, gutenberg, stopwords


class Vocabulary:

    def __init__(self):
        self.vocab_dict = {}

    def increase_by_one(self, token):
        if not token in self.vocab_dict:
            self.vocab_dict[token] = 1
        else:
            self.vocab_dict[token] += 1

    def keep_most(self, n):

        sorted_vocab = [k for k, v in sorted(self.vocab_dict.items(), key=lambda item: item[1])]
        if n> len(sorted_vocab):
            return sorted_vocab
        else:
            keep_list = sorted_vocab[-(n-1):]
            return keep_list


def prepare_guttenberg():

    books = gutenberg.fileids()
    stop = stopwords.words()
    vocabulary = Vocabulary()

    for book_name in books:
        book = gutenberg.sents(book_name)
        count = 0
        with open('data/nltk/' + book_name, 'w', encoding='utf-8') as f:
            for sent in book:
                for word in sent:
                    f.write(word.lower() + ' ')
                    if not word.lower() in stop:
                        vocabulary.increase_by_one(word.lower())

                count += 1
                if count == 10:
                    f.write('\n')
                    count = 0

    vocab_list = vocabulary.keep_most(100000)
    # write the vocabulary
    with open(os.path.join('data/nltk', 'vocabulary'), 'w', encoding='utf-8') as f:
        f.write('<unk>\n')
        for token in vocab_list:
            f.write(token + '\n')

def prepare_reuters():

    stop = stopwords.words()
    vocabulary = Vocabulary()

    file_ids = reuters.fileids()
    all_categories = reuters.categories()
    print(len(all_categories))

    for file_id in file_ids:
        categories = reuters.categories(file_id)
        labels = []
        for category in categories:
            labels.append(all_categories.index(category))

        words = reuters.words(file_id)
        with open('data/reuters/' + file_id, 'w', encoding='utf-8') as f:
            for word in words:
                f.write(word.lower() + ' ')
                if not word.lower() in stop:
                    vocabulary.increase_by_one(word.lower())
            f.write('\t')
            for label in labels:
                f.write(str(label) + ' ')

    vocab_list = vocabulary.keep_most(100000)
    # write the vocabulary
    with open(os.path.join('data/reuters', 'vocabulary'), 'w', encoding='utf-8') as f:
        f.write('<unk>\n')
        for token in vocab_list:
            f.write(token + '\n')

if __name__=='__main__':

    #prepare_nltk_corpora()
    #prepare_reuters()
    #exit()
    corpus_name = sys.argv[1]
    output_filename = sys.argv[2]
    vocab_size = int(sys.argv[3])

    count = 0
    for filename in os.listdir('data/reddit_small'):

        if filename == 'vocabulary':
            continue

        corpus = Corpus(filename=download('subreddit-' + filename))
        vocabulary = Vocabulary()

        nlp = spacy.load('en_core_web_sm')

        for convo in corpus.iter_conversations():
            title = convo.meta['title']
            subreddit = convo.meta['subreddit']
            # this gives us the text of the post
            text = convo.get_utterance(convo.get_chronological_utterance_list()[0].conversation_id).text
            if text == '' or text == '[deleted]' or text == '[removed]':
                continue
            else:
                post = title + '\t' + text
                post = post.replace('\n', '').lower()
                doc = nlp(post)
                doc_token_list = []
                for token in doc:
                    if not token.is_stop and token.is_alpha:
                        vocabulary.increase_by_one(token.lemma_)
                        doc_token_list.append(token.lemma_)

                # write the document
                with open(os.path.join(output_filename, subreddit), 'a', encoding='utf-8') as f:
                    for token in doc_token_list:
                        f.write(token + ' ')
                    f.write('\n')

        vocab_list = vocabulary.keep_most(vocab_size)
        # write the vocabulary
        with open(os.path.join(output_filename, 'vocabulary'), 'w', encoding='utf-8') as f:
            f.write('<unk>\n')
            for token in vocab_list:
                f.write(token + '\n')

        count += 1


