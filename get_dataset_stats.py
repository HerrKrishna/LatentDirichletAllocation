import sys
import os
from collections import Counter

if __name__ == '__main__':

    corpus_dir = sys.argv[1]
    doc_count = 0
    counter = Counter()
    for filename in os.listdir(corpus_dir):
        if filename == 'vocabulary':
            continue
        with open(corpus_dir + '/' + filename, 'r') as f:
            lines = f.readlines()
        doc_count += len(lines)
        counter.update({filename:len(lines)})

    print(doc_count/70)

