import pickle as cPickle
from model import Model


import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")

FLR = False
FLS = False


def evaluate(x, num_classes = 2, k_fold = 10):
    revs, embedding, W2, word_idx_map, vocab, max_l = x[0], x[1], x[2], x[3], x[4], x[5]
    if FLR:
        embedding = W2
    embedding_dim = 200
    vocab_size = len(vocab) + 1
    filter_sizes = [3, 4, 5]
    num_filters = 100
    vector_length = max_l + 2 * 4
    cnn_model = Model()
    trainable = not FLS
    print(FLR,trainable)
    cnn_model.build_model(embedding_dim, vocab_size, filter_sizes, num_filters, vector_length, num_classes, trainable)
    cnn_model.run(revs, embedding, word_idx_map, max_l, k_fold)


def evaluate_tweets():
    x = cPickle.load(open("twitter.p", "rb"))
    evaluate(x, 2, 1)

if __name__=="__main__":

   evaluate_tweets()




