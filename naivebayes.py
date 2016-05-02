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
    vec_of_word_count_c1 = ones(num_of_word)  # p(w|c_1) = p(w_1|c_1)...p(w_j|c_1)
    vec_of_word_count_c0 = ones(num_of_word)
    total_word_count_c1 = 2.0
    total_word_count_c0 = 2.0
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


def naive_bayes_classify(vec_to_classify, prob_vector_c1, prob_vector_c0, prior_c1):
    log_prob_vector_c1 = log(prob_vector_c1)
    log_prob_vector_c0 = log(prob_vector_c0)
    log_posterior_c1 = sum(log_prob_vector_c1 * vec_to_classify) + log(prior_c1)  # log(p(c_1|w)) = log(p(w|c_1)p(c_1))
    log_posterior_c0 = sum(log_prob_vector_c0 * vec_to_classify) + log(1.0 - prior_c1)  # log(p(c_0|w)) = log(p(w|c_0)p(c_0))
    if log_posterior_c1 > log_posterior_c0:
        return 1
    else:
        return 0


def test_naive_bayes():
    dataset_list, class_vector = load_datasets()
    voca_dictionary = create_voca_dictionary(dataset_list)
    training_mat = []
    for dataset in dataset_list:
        training_mat.append(dataset_to_feat_vector(voca_dictionary, dataset))
    prob_vector_c0, prob_vector_c1, prior_c1 = training_naive_bayes(training_mat, class_vector)
    test_vector_1 = ['love', 'my', 'dalmation']  # test 1
    test_feat_vec_1 = dataset_to_feat_vector(voca_dictionary, test_vector_1)
    print test_vector_1, 'classified as : ', naive_bayes_classify(test_feat_vec_1, prob_vector_c1, prob_vector_c0, prior_c1)
    test_vector_2 = ['stupid', 'garbage']  # test 2
    test_feat_vec_2 = dataset_to_feat_vector(voca_dictionary, test_vector_2)
    print test_vector_2, 'classified as : ', naive_bayes_classify(test_feat_vec_2, prob_vector_c1, prob_vector_c0, prior_c1)


def text_parser(text):
    import re
    list_of_tokens = re.split(r'\W*', text)
    return [token.lower() for token in list_of_tokens if len(token) > 2]


def verify_classifier():
    dataset_list = []
    class_list = []  # read data
    for i in range(1, 26):
        word_list = text_parser(open('email/spam/%d.txt' % i).read())
        dataset_list.append(word_list)
        class_list.append(1)
        word_list = text_parser(open('email/ham/%d.txt' % i).read())
        dataset_list.append(word_list)
        class_list.append(0)
    voca_dictionary = create_voca_dictionary(dataset_list)
    training_set = range(50)
    test_set = []  # divide training set and test set
    for i in range(10):
        rand_idx = int(random.uniform(0, len(training_set)))
        test_set.append(rand_idx)
        del (training_set[rand_idx])
    training_matrix = []
    training_class = []
    for i in training_set:  # training set
        training_matrix.append(dataset_to_feat_vector(voca_dictionary, dataset_list[i]))
        training_class.append(class_list[i])
    prob_vector_c0, prob_vector_c1, prior_c1 = training_naive_bayes(training_matrix, training_class)
    error_count = 0.0
    for i in test_set:  # verify
        test_feat_vector = dataset_to_feat_vector(voca_dictionary, dataset_list[i])
        classified_value = naive_bayes_classify(test_feat_vector, prob_vector_c1, prob_vector_c0, prior_c1)
        if classified_value != class_list[i]:
            error_count += 1
    print 'the error rate is: ', float(error_count) / len(test_set)