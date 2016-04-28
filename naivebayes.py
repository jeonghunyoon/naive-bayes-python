from numpy import *

'''
sample datasets
'''


def load_datasets():
    dataset_list = [['my', 'dog', 'has', 'flea', 'problem', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vector = [0, 1, 0, 1, 0, 1]
    return dataset_list, class_vector


def create_voca_dictionary(dataset_list):
    voca_set = set([])
    for dataset in dataset_list:
        voca_set = voca_set | set(dataset)
    return voca_set


def dataset_to_feat_vector(voca_dictionary, dataset):
    feat_vector = zeros(voca_dictionary)
    for word in dataset:
        if word in voca_dictionary:
            feat_vector[voca_dictionary.index(word)] = 1
        else:
            print "the word : %s is niot in my vocabulary dictionary." % word
    return feat_vector
