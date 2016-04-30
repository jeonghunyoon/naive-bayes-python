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
    return list(voca_set)


def dataset_to_feat_vector(voca_dictionary, dataset):
    feat_vector = [0] * (len(voca_dictionary))
    for word in dataset:
        if word in voca_dictionary:
            feat_vector[voca_dictionary.index(word)] = 1
        else:
            print "the word : %s is niot in my vocabulary dictionary." % word
    return feat_vector


def training_naive_bayes(training_matrix, training_category):
    num_of_training_set = len(training_matrix)
    num_of_word = len(training_matrix[0])
    prior_c1 = sum(training_category) / float(num_of_training_set)
    vec_of_word_count_c1 = zeros(num_of_word)  # p(w|c_1) = p(w_1|c_1)...p(w_j|c_1)
    vec_of_word_count_c0 = zeros(num_of_word)
    total_word_count_c1 = 0.0
    total_word_count_c0 = 0.0
    for i in range(num_of_training_set):
        if training_category[i] == 1:
            vec_of_word_count_c1 += training_matrix[i]
            total_word_count_c1 += sum(training_matrix[i])
        else:
            vec_of_word_count_c0 += training_matrix[i]
            total_word_count_c0 += sum(training_matrix[i])
    prob_vector_c1 = vec_of_word_count_c1 / total_word_count_c1
    prob_vector_c0 = vec_of_word_count_c0 / total_word_count_c0
    return prob_vector_c0, prob_vector_c1, prior_c1
