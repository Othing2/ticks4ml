# coding=utf-8

"""
auth: bill guan
time: 2018-3-18
desc: 语料路径配置信息
"""
# 原始语料目录
people_dir = "/home/worker/backup/people_2014/people"
# 语料文件
txt_path = "/home/worker/backup/people_2014/pre_chars_for_w2v.txt"
# 语料词频文件
vec_path = "/home/worker/backup/people_2014/pre_vocab.txt"
# 处理低频字的语料
unk_path = "/home/worker/backup/people_2014/chars_for_w2v.txt"
# word2vec 生成的字向量文件
char_vec_path = "/home/worker/backup/people_2014/char_vec.txt"
# 语料的标注[0/S 1/B 2/M 3/E]文件 每行的前80个数字表示字在向量文件中的索引,后80个数字表示对应的标注
char_tag_path = "/home/worker/backup/people_2014/char_tag.txt"
# 分词训练语料
char_train_path = "/home/worker/backup/people_2014/char_train.txt"
# 分词测试语料
char_test_path = "/home/worker/backup/people_2014/char_test.txt"
