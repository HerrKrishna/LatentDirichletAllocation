import os
from collections import Counter

tokens = []
for filename in os.listdir('data/reddit_big'):
    with open('data/reddit_big/'+filename, 'r')as f:
        lines = f.readlines()
    for line in lines:
        tokens+= line.strip('\n').split(' ')

counter = Counter(tokens)
vocab = [item[0] for item in counter.most_common(100000)]

with open('data/reddit_big/vocabulary', 'w') as f:
    for word in vocab:
        f.write(word + '\n')

