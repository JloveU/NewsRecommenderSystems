#!python3
"""
Created on 2015-12-24
@author: yuqiang
Evaluation of NMF-User-based Collaborative Filtering
"""

import numpy
import data

user_news_array, user_ids, news_ids = data.get_user_news_array()
user_num = len(user_ids)
news_num = len(news_ids)
news_dict = data.get_news_dict()
user_news_recommend_indexes = numpy.load("user_news_recommend_indexes.npy")

for i in range(user_num):
    clicked_news_ids = news_ids[user_news_array[i] == 1]
    recommend_news_ids = news_ids[user_news_recommend_indexes[i]]
    clicked_news_contents = [news_dict[key] for key in clicked_news_ids]
    recommend_news_contents = [news_dict[key] for key in recommend_news_ids]

print("")

