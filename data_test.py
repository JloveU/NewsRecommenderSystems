#!python3
"""
Created on 2015-12-12
@author: yuqiang
Test of given data of news
"""

# read data from file
f = open("user_click_data.txt", "r", 1, "utf-8")
user_ids = []
news_ids = []
visit_times = []
news_titles = []
news_bodies = []
while True:
    line = f.readline()
    if line:
        p = line.split('\t')
        user_ids.append(int(p[0]))
        news_ids.append(int(p[1]))
        visit_times.append(int(p[2]))
        news_titles.append(p[3])
        news_bodies.append(p[4])
    else:
        break
f.close()

print("total lines: %d" % len(user_ids))

# remove duplicate to calculate number of user and news
unique_user_ids = list(set(user_ids))
# unique_user_ids.sort(user_ids.index)
unique_news_ids = list(set(news_ids))

print("")
print("unique user number: %d" % len(unique_user_ids))
print("unique news number: %d" % len(unique_news_ids))

# calculate the threshold of visit times to divide data into a train set and a test set
# visit_times.sort()
time_start = min(visit_times)
time_end = max(visit_times)
time_threshold = time_start + (time_end - time_start) * 2 // 3
train_set_indexes = [index for (index, value) in enumerate(visit_times) if value < time_threshold]
test_set_indexes = [index for (index, value) in enumerate(visit_times) if value >= time_threshold]

print("")
print("start     time: %d" % time_start)
print("end       time: %d" % time_end)
print("threshold time: %d (early 2/3 of time period)" % time_threshold)
print("train set size: %d (before threshold time)" % len(train_set_indexes))
print("test  set size: %d (after  threshold time)" % len(test_set_indexes))
