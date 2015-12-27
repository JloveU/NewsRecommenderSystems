#!python3
"""
Created on 2015-12-13
@author: yuqiang
NMF-User-based Collaborative Filtering
"""

import time
import numpy
import nmf
import data
import progress

"""
# get user-news array
user_news_array, user_ids, news_ids = data.get_user_news_array()
user_num = len(user_ids)
news_num = len(news_ids)

# NMF
V = numpy.float16(user_news_array.T)
N = len(V)
M = len(V[0])
K = 15  # TODO refine this parameter
W_init = numpy.random.rand(N, K)
H_init = numpy.random.rand(K, M)
W, H = nmf.nmf(V, W_init, H_init)
del V, W_init, H_init, W, M, N
# estimatedV = numpy.dot(W, H)

# calculate similarity between users
print("Calculate similarity between users started.")
time_start = time.time()
user_user_similarities = numpy.zeros((user_num, user_num), numpy.float16)
H_norm = numpy.power(H, 2)
H_norm = H_norm.sum(0)
H_norm = numpy.sqrt(H_norm)  # norm of each column vector in H
H_norm = numpy.tile(H_norm, (K, 1))
H_normalized = H / H_norm
H_normalized_transpose = H_normalized.transpose()
computed_count = 0
compute_step = 1000  # to avoid MemoryError, only compute a part each time
while computed_count < user_num:
    compute_upper_limit = min((computed_count + compute_step, user_num))
    user_user_similarities[computed_count:compute_upper_limit, :] = numpy.dot(H_normalized_transpose[computed_count:compute_upper_limit, :], H_normalized)
    computed_count += compute_step
del H, H_norm, H_normalized, H_normalized_transpose, computed_count, compute_step
time_end = time.time()
print("Calculate similarity between users ended. %f s cost." % (time_end - time_start))

# find k nearest neighbors of users
print("Find k nearest neighbors of users started.")
time_start = time.time()
neighbor_size = 20  # TODO refine this parameter
user_neighbors_indexes = numpy.zeros((user_num, neighbor_size), numpy.int16)
inverse_indexes = range(user_num-2, user_num-neighbor_size-2, -1)  # choose the last k in the sorted list, remove the last one which is oneself
for i in range(user_num):
    sorted_indexes = numpy.argsort(user_user_similarities[i, :])
    user_neighbors_indexes[i, :] = sorted_indexes[inverse_indexes]
    if i % 100 == 0:
        print("%.1f%%" % (i / user_num * 100))
del inverse_indexes
time_end = time.time()
print("Find k nearest neighbors of users ended. %f s cost." % (time_end - time_start))

# predict ratings
print("Predict ratings started.")
time_start = time.time()
user_news_array = numpy.int8(user_news_array)  # int is faster than bool_
user_news_predict_array = numpy.zeros((user_num, news_num), numpy.float16)
eps = numpy.finfo(float).eps
for i in range(user_num):
    similarities_sum = 0.0
    for user_neighbors_index in user_neighbors_indexes[i]:
        user_news_predict_array[i] += user_news_array[user_neighbors_index] * user_user_similarities[user_neighbors_index, i]
        similarities_sum += user_user_similarities[user_neighbors_index, i]
    user_news_predict_array[i] /= (similarities_sum + eps)
    if i % 100 == 0:
        print("%.1f%%" % (i / user_num * 100))
user_news_predict_array[user_news_array == 1] = 0  # remove news one user has clicked
del eps
time_end = time.time()
print("Predict ratings ended. %f s cost." % (time_end - time_start))

# choose first k news to recommend to users
print("Choose first k news to recommend to users started.")
time_start = time.time()
recommend_size = 10  # TODO refine this parameter
user_news_recommend_indexes = numpy.zeros((user_num, recommend_size), numpy.int16)
inverse_indexes = range(news_num-1, news_num-recommend_size-1, -1)  # choose the last k in the sorted list
for i in range(user_num):
    sorted_indexes = numpy.argsort(user_news_predict_array[i, :])
    user_news_recommend_indexes[i, :] = sorted_indexes[inverse_indexes]
    if i % 100 == 0:
        print("%.1f%%" % (i / user_num * 100))
del inverse_indexes
time_end = time.time()
print("Choose first k news to recommend to users ended. %f s cost." % (time_end - time_start))

print("")
"""


def train():
    """Train with the train set

    Train with the train set, and return the user_user_similarities array.

    Returns:
        user_user_similarities(Type: numpy.ndarray): The similarities between user and user(user_user_similarities[i, j] represents the similarity between user "i" and user "j". Similarity of oneself is "1".)
        user_ids(Type: numpy.ndarray(vector)): user's ids(from small to large) associated with the array
    """

    # get user-news array
    user_news_array_of_train = numpy.load("user_news_array_of_train.npy")
    user_ids_of_train = numpy.load("user_ids_of_train.npy")
    user_news_array = user_news_array_of_train
    user_ids = user_ids_of_train
    user_num = len(user_ids)
    del user_news_array_of_train, user_ids_of_train

    # NMF
    V = numpy.float16(user_news_array.T)
    K = 15  # TODO refine this parameter
    W, H = nmf.nmf(V, K)
    del V, W
    # estimatedV = numpy.dot(W, H)

    # calculate similarity between users
    print("Calculate similarity between users started.")
    time_start = time.time()
    user_user_similarities = numpy.zeros((user_num, user_num), numpy.float16)
    H_norm = numpy.power(H, 2)
    H_norm = H_norm.sum(0)
    H_norm = numpy.sqrt(H_norm)  # norm of each column vector in H
    H_norm = numpy.tile(H_norm, (K, 1))
    eps = numpy.finfo(float).eps
    H_normalized = H / (H_norm + eps)
    H_normalized_transpose = H_normalized.transpose()
    computed_count = 0
    compute_step = 1000  # to avoid MemoryError, only compute a part each time
    while computed_count < user_num:
        compute_upper_limit = min((computed_count + compute_step, user_num))
        user_user_similarities[computed_count:compute_upper_limit, :] = numpy.dot(H_normalized_transpose[computed_count:compute_upper_limit, :], H_normalized)
        computed_count += compute_step
    del H, H_norm, H_normalized, H_normalized_transpose, computed_count, compute_step
    time_end = time.time()
    print("Calculate similarity between users ended. %f s cost." % (time_end - time_start))

    print("[NMF-User-based Collaborative Filtering] Train finished!")

    return user_user_similarities, user_ids


def recommend(user_user_similarities, user_ids):
    """Recommend with the test set

    Recommend with the test set, and return the user_news_rating_predictions array.

    Args:
        user_user_similarities(Type: numpy.ndarray): The similarities between user and user(user_user_similarities[i, j] represents the similarity between user "i" and user "j". Similarity of oneself is "1".)
        user_ids(Type: numpy.ndarray(vector)): user's ids(from small to large) associated with the array

    Returns:
        user_news_rating_predictions(Type: numpy.ndarray): The rating prediction of each user to each news(rating_prediction[i, j] represents the rating prediction of user "i" to news "j".)
        user_ids(Type: numpy.ndarray(vector)): user's ids(from small to large) associated with the array
        news_ids(Type: numpy.ndarray(vector)): news's ids(from small to large) associated with the array
    """

    # get user-news array
    user_news_array_of_train = numpy.load("user_news_array_of_train.npy")
    user_ids_of_train = numpy.load("user_ids_of_train.npy")
    news_ids_of_train = numpy.load("news_ids_of_train.npy")
    user_news_array_of_test = numpy.load("user_news_array_of_test.npy")
    user_ids_of_test = numpy.load("user_ids_of_test.npy")
    news_ids_of_test = numpy.load("news_ids_of_test.npy")
    user_num_of_train = len(user_ids_of_train)
    news_num_of_train = len(news_ids_of_train)
    user_num_of_test = len(user_ids_of_test)
    news_num_of_test = len(news_ids_of_test)

    # find k nearest neighbors of users
    print("Find k nearest neighbors of users started.")
    time_start = time.time()
    neighbor_size = min(20, user_num_of_train-1)  # TODO refine this parameter
    user_neighbors_indexes = numpy.zeros((user_num_of_test, neighbor_size), numpy.int16)
    user_ids_of_train_dict = {user_id: index for (index, user_id) in enumerate(user_ids_of_train)}
    user_index_from_test_to_train_dict = {user_index_in_test: user_ids_of_train_dict[user_id_of_test] for (user_index_in_test, user_id_of_test) in enumerate(user_ids_of_test)}  # dictionary of user index from test to train
    for i in range(user_num_of_test):
        sorted_indexes = numpy.argsort(-user_user_similarities[user_index_from_test_to_train_dict[i], :])
        user_neighbors_indexes[i, :] = sorted_indexes[1:neighbor_size+1]  # choose the first k in the sorted list, remove the first one which is oneself
        if i % 100 == 0:
            # print("%.1f%%" % (i / user_num_of_test * 100))
            progress.update(i / user_num_of_test)
    progress.update(1)
    time_end = time.time()
    print("Find k nearest neighbors of users ended. %f s cost." % (time_end - time_start))

    # predict ratings
    print("Predict ratings started.")
    time_start = time.time()
    news_ids_of_test_dict = {news_id: index for (index, news_id) in enumerate(news_ids_of_test)}
    news_index_from_train_to_test_dict = {news_index_in_train: news_ids_of_test_dict[news_id_of_train] for (news_index_in_train, news_id_of_train) in enumerate(news_ids_of_train)}  # dictionary of news index from train to test
    user_news_array_of_train_expanded = numpy.zeros((user_num_of_train, news_num_of_test), numpy.int8)  # int is faster than bool_
    user_news_array_of_train_expanded[:, [news_index_from_train_to_test_dict[i] for i in range(news_num_of_train)]] = user_news_array_of_train  # expand the column of user_news_array_of_train to the same size as user_news_array_of_test
    user_news_rating_predictions = numpy.zeros((user_num_of_test, news_num_of_test), numpy.float16)
    eps = numpy.finfo(float).eps
    for i in range(user_num_of_test):
        this_user_index_in_test = user_index_from_test_to_train_dict[i]
        similarities_sum = 0.0
        for user_neighbors_index in user_neighbors_indexes[i]:
            user_news_rating_predictions[i] += user_news_array_of_train_expanded[user_neighbors_index] * user_user_similarities[user_neighbors_index, this_user_index_in_test]
            similarities_sum += user_user_similarities[user_neighbors_index, this_user_index_in_test]
        user_news_rating_predictions[i] /= (similarities_sum + eps)
        if i % 100 == 0:
            # print("%.1f%%" % (i / user_num_of_test * 100))
            progress.update(i / user_num_of_test)
    progress.update(1)
    user_news_rating_predictions[user_news_array_of_train_expanded[[user_index_from_test_to_train_dict[i] for i in range(user_num_of_test)], :] == 1] = 0  # remove news one user has clicked
    time_end = time.time()
    print("Predict ratings ended. %f s cost." % (time_end - time_start))

    print("[NMF-User-based Collaborative Filtering] Recommend finished!")

    return user_news_rating_predictions, user_ids_of_test, news_ids_of_test


user_user_similarities, user_ids_of_train = train()
user_news_rating_predictions, user_ids_of_test, news_ids_of_test = recommend(user_user_similarities, user_ids_of_train)
user_news_array_of_test = numpy.load("user_news_array_of_test.npy")
import scipy.io
scipy.io.savemat("data_for_evaluation.mat",
                 {"user_news_rating_predictions": user_news_rating_predictions,
                  "user_ids": user_ids_of_test,
                  "news_ids": news_ids_of_test,
                  "user_news_array_of_test": user_news_array_of_test})
