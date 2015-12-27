#!python3
"""
Created on 2015-12-13
@author: yuqiang
Read of given data and reform it to useful format
"""

import time
import numpy


def _form_user_news_array_from_list(user_news_pairs):
    """Form user-news array from list(private method)

    Form the user-news array from given list, whose rows represent users and cols represent newses.
    The meaning of A[i, j] in the result array is whether user "i" clicked news "j"(if true, 1, else, 0).

    Args:
        user_news_pairs(Type: list[(user_id, news_id)]): list of user-news pair which represents the user clicked the news

    Returns:
        user_news_array(Type: numpy.ndarray): The result user-news array(rows for users and cols for newses)
        user_ids(Type: numpy.ndarray(vector)): user's ids(from small to large) associated with the array
        news_ids(Type: numpy.ndarray(vector)): news's ids(from small to large) associated with the array
    """

    user_ids = [user_id for (user_id, news_id) in user_news_pairs]
    news_ids = [news_id for (user_id, news_id) in user_news_pairs]

    # unique and sort user and news
    user_ids = numpy.array(list(set(user_ids)))
    user_ids.sort()
    news_ids = numpy.array(list(set(news_ids)))
    news_ids.sort()

    # form the array
    user_num = len(user_ids)
    news_num = len(news_ids)
    user_ids_dict = {value: index for (index, value) in enumerate(user_ids)}
    news_ids_dict = {value: index for (index, value) in enumerate(news_ids)}
    user_news_array = numpy.zeros((user_num, news_num), numpy.bool_)
    for (user_id, news_id) in user_news_pairs:
        user_news_array[user_ids_dict[user_id], news_ids_dict[news_id]] = 1

    return user_news_array, user_ids, news_ids


def get_user_news_array():
    """Form user-news array

    Form the user-news array, whose rows represent users and cols represent newses.
    The meaning of A[i, j] in the result array is whether user "i" clicked news "j"(if true, 1, else, 0).

    Returns:
        user_news_array(Type: numpy.ndarray): The result user-news array(rows for users and cols for newses)
        user_ids(Type: numpy.ndarray(vector)): user's ids(from small to large) associated with the array
        news_ids(Type: numpy.ndarray(vector)): news's ids(from small to large) associated with the array
    """

    # read data from file
    f = open("user_click_data.txt", "r", 1, "utf-8")
    user_news_pairs = []
    while True:
        line = f.readline()
        if line:
            p = line.split('\t')
            user_news_pairs.append((int(p[0]), int(p[1])))
        else:
            break
    f.close()

    return _form_user_news_array_from_list(user_news_pairs)


def get_user_news_arrays_of_train_and_test(remove_new_users_in_test=False, remove_new_newses_in_test=False):
    """Form user-news arrays of train and test

    Form the user-news array of train and test, whose rows represent users and cols represent newses.
    The meaning of A[i, j] in the result array is whether user "i" clicked news "j"(if true, 1, else, 0).

    Args:
        remove_new_users_in_test(Type: bool): whether to remove new users in test
        remove_new_newses_in_test(Type: bool): whether to remove new newses in test

    Returns:
        user_news_array_of_train(Type: numpy.ndarray): The result user-news array of train(rows for users and cols for newses)
        user_ids_of_train(Type: numpy.ndarray(vector)): user's ids(from small to large) associated with the user-news array of train
        news_ids_of_train(Type: numpy.ndarray(vector)): news's ids(from small to large) associated with the user-news array of train
        user_news_array_of_test(Type: numpy.ndarray): The result user-news array of test(rows for users and cols for newses)
        user_ids_of_test(Type: numpy.ndarray(vector)): user's ids(from small to large) associated with the user-news array of test
        news_ids_of_test(Type: numpy.ndarray(vector)): news's ids(from small to large) associated with the user-news array of test
    """

    # read data from file
    f = open("user_click_data.txt", "r", 1, "utf-8")
    user_news_time_list = []
    while True:
        line = f.readline()
        if line:
            p = line.split('\t')
            user_news_time_list.append((int(p[0]), int(p[1]), int(p[2])))
        else:
            break
    f.close()

    # calculate the threshold of visit times to divide data into a train set and a test set
    click_times = [click_time for (user_id, news_id, click_time) in user_news_time_list]
    time_start = min(click_times)
    time_end = max(click_times)
    time_threshold = time_start + (time_end - time_start) * 2 // 3

    # divide data into train set and test set
    user_news_pairs_of_all = [(user_id, news_id) for (user_id, news_id, click_time) in user_news_time_list]
    user_news_pairs_of_train = [(user_id, news_id) for (user_id, news_id, click_time) in user_news_time_list if click_time < time_threshold]
    user_news_pairs_of_test = [(user_id, news_id) for (user_id, news_id, click_time) in user_news_time_list if click_time >= time_threshold]

    # form the train array
    user_news_array_of_all, user_ids_of_all, news_ids_of_all = _form_user_news_array_from_list(user_news_pairs_of_all)
    user_news_array_of_train, user_ids_of_train, news_ids_of_train = _form_user_news_array_from_list(user_news_pairs_of_train)
    # user_news_array_of_test, user_ids_of_test, news_ids_of_test = _form_user_news_array_from_list(user_news_pairs_of_test)

    # form the test array(remove new users or new newses)
    user_ids_of_all_dict = {user_id: index for (index, user_id) in enumerate(user_ids_of_all)}
    news_ids_of_all_dict = {news_id: index for (index, news_id) in enumerate(news_ids_of_all)}
    train_samples = numpy.array([[user_ids_of_all_dict[user_id], news_ids_of_all_dict[news_id]] for (user_id, news_id) in user_news_pairs_of_train]).transpose()  # coordinates of train samples in user_news_array_of_all
    user_news_array_of_test = user_news_array_of_all.copy()
    user_news_array_of_test[train_samples[0, :], train_samples[1, :]] = 0  # remove train samples in user_news_array_of_all
    user_indices_of_train_in_all = numpy.array([index for (index, user_id) in enumerate(user_ids_of_all) if user_id in user_ids_of_train])  # indices of users in user_ids_of_all who appeared in train set
    user_indices_of_train_in_all.reshape((-1, 1))  # reshape to column vector
    news_indices_of_train_in_all = numpy.array([index for (index, news_id) in enumerate(news_ids_of_all) if news_id in news_ids_of_train])  # indices of newses in news_ids_of_all who appeared in train set
    news_indices_of_train_in_all.reshape((1, -1))  # reshape to row vector
    user_num = len(user_ids_of_all)
    news_num = len(news_ids_of_all)
    user_news_array_of_test = user_news_array_of_test[user_indices_of_train_in_all if remove_new_users_in_test else range(user_num), :]  # remove new users
    user_news_array_of_test = user_news_array_of_test[:, news_indices_of_train_in_all if remove_new_newses_in_test else range(news_num)]  # remove new newses
    user_ids_of_test = user_ids_of_all[user_indices_of_train_in_all if remove_new_users_in_test else range(user_num)]  # remove new users
    news_ids_of_test = news_ids_of_all[news_indices_of_train_in_all if remove_new_newses_in_test else range(news_num)]  # remove new newses

    # remove users who don't appear in test set
    hold_indices = [index for (index, user_id) in enumerate(user_ids_of_test) if user_id in [user_id for (user_id, news_id) in user_news_pairs_of_test]]
    if len(hold_indices) > 0:
        user_news_array_of_test = user_news_array_of_test[hold_indices, :]
        user_ids_of_test = user_ids_of_test[hold_indices]

    # write data to files
    numpy.save("user_news_array_of_train.npy", user_news_array_of_train)
    numpy.save("user_ids_of_train.npy", user_ids_of_train)
    numpy.save("news_ids_of_train.npy", news_ids_of_train)
    numpy.save("user_news_array_of_test.npy", user_news_array_of_test)
    numpy.save("user_ids_of_test.npy", user_ids_of_test)
    numpy.save("news_ids_of_test.npy", news_ids_of_test)

    return user_news_array_of_train, user_ids_of_train, news_ids_of_train, user_news_array_of_test, user_ids_of_test, news_ids_of_test


def get_news_dict():
    """Form news dict

    Form the news dict, whose key is news id and value is news title and content.

    Returns:
        news_dict(Type: dictionary): key: news id, value: news title and content
    """

    # read data from file
    f = open("user_click_data.txt", "r", 1, "utf-8")
    news_dict = {}
    while True:
        line = f.readline()
        if line:
            p = line.split('\t')
            news_dict.setdefault(int(p[1]), p[3] + p[4])
        else:
            break
    f.close()

    return news_dict


def get_user_clicked_news_dict():
    """Form user clicked news dict

    Form the user clicked news dict, whose key is user id and value is news ids the user has clicked.

    Returns:
        user_clicked_news_dict(Type: dictionary): key: user id, value: news ids the user has clicked
    """

    user_news_array, user_ids, news_ids = get_user_news_array()
    user_num = len(user_ids)
    user_clicked_news_dict = {}
    for i in range(user_num):
        clicked_news_ids = news_ids[user_news_array[i] == 1]
        user_clicked_news_dict.setdefault(user_ids[i], clicked_news_ids)

    return user_clicked_news_dict
