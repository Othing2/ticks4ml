# -*- coding: utf-8 -*-

"""
auth: Bill Guan
date: 2018-3-18
desc: 生成语料分词用的训练集和测试集, 语料标注信息文件请看generate_char_training.py 文件
"""
import os, sys
import random


def main(argc, argv):
    if argc < 2:
        print("Usage:%s <input>" % (argv[0]))
        sys.exit(1)
    SENTENCE_LEN = 80
    fp = open(argv[1], "r")
    nl = 0
    bad = 0
    test = 0
    tr_p = open(argv[2], "w")  # 训练文件
    te_p = open(argv[3], "w")  # 测试文件
    while True:
        line = fp.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        ss = line.split(' ')

        if len(ss) != (2 * SENTENCE_LEN):
            print("len is:%d" % (len(ss)))
            continue
        numV = 0
        for i in range(SENTENCE_LEN):
            if int(ss[i]) != 0:
                numV += 1
                if numV > 2:
                    break
        if numV <= 2:
            bad += 1
        else:
            r = random.random()
            if r <= 0.02 and test < 8000:
                te_p.write("%s\n" % (line))
                test += 1
            else:
                tr_p.write("%s\n" % (line))
        nl += 1
    fp.close()
    print("got bad:%d" % (bad))


if __name__ == '__main__':
    from _config import *
    argv = ["", char_tag_path, char_train_path, char_test_path]
    main(len(argv), argv)
