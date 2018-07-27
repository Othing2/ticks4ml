# coding=utf-8

"""
加载数据
"""

import numpy as np


def load_w2v(path, expectDim):
    fp = open(path, "r")
    print("load data from:", path)
    line = fp.readline().strip()
    ss = line.split(" ")
    total = int(ss[0])
    dim = int(ss[1])
    assert (dim == expectDim)
    ws = []
    mv = [0 for i in range(dim)]
    second = -1
    for t in range(total):
        if ss[0] == '<UNK>':
            second = t
        line = fp.readline().strip()
        ss = line.split(" ")
        assert (len(ss) == (dim + 1))
        vals = []
        for i in range(1, dim + 1):
            fv = float(ss[i])
            mv[i - 1] += fv
            vals.append(fv)
        ws.append(vals)
    for i in range(dim):
        mv[i] = mv[i] / total
    assert (second != -1)
    # append one more token , maybe useless
    ws.append(mv)
    if second != 1:
        t = ws[1]
        ws[1] = ws[second]
        ws[second] = t
    fp.close()
    return np.asarray(ws, dtype=np.float32)


def load_word_file(path, w2v, bath_size=0, bath_id=0):
    """加载(训练|测试)文件"""
    # [0/S 1/B 2/M 3/E]
    w_dim = w2v.WordDim
    max_len = 80
    with open(path, 'r') as f:
        word_tags = []
        if bath_size > 0:
            lines = f.readlines()[bath_size*bath_id:bath_size*(bath_id+1)]
        else:
            lines = f.readlines()
        for line in lines:
            ln = map(lambda x: int(x), line.strip().split(' '))
            word = filter(lambda x: x != 0, ln[:max_len])
            tag = ln[max_len:max_len+len(word)]

            assert len(word) == len(tag)
            vect_x = np.zeros([w_dim, len(word)])
            vect_y = np.zeros([4, len(word)])
            for i, w in enumerate(word):
                v = w2v.GetVectorByIndex(w)
                if not v:
                    print("Not exit", w)
                    continue
                vect_x[:, i] = v
                vect_y[tag[i], i] = 1.0
            word_tags.append((vect_x, vect_y))
    return word_tags


def word_pretty_display(path, w2v, ll=0):
    """显示(训练|测试)文件中的一行中文的 分词结果"""
    tag_dict = {0: 'S', 1: 'B', 2: 'M', 3: 'E'}
    max_len = 80
    with open(path, 'r') as f:
        line = f.readline()
        if ll:
            line = f.readlines()[:ll][-1]
        ln = map(lambda x: int(x), line.strip().split(' '))
        word = filter(lambda x: x != 0, ln[:max_len])
        tag = ln[max_len:max_len + len(word)]

        words = [w2v.GetWordByIndex(w) for w in word]
        tags = ''.join([tag_dict.get(t) for t in tag])
        lt = len(words)
        for i, t in enumerate(tags[::-1]):
            if t == 'S' or t == 'E':
                if i == 0:
                    words.append('')
                else:
                    words.insert(lt-i, '/')

    return u''.join(words)


if __name__ == '__main__':
    from refers.tag_build._config import *
    from refers.tag_build.w2v import Word2vecVocab

    w2v = Word2vecVocab()
    w2v.Load(char_vec_path)

    ws = word_pretty_display(char_train_path, w2v, ll=1)
    print(ws)

    # batch_size = 100
    # ll = 0
    # train_word = load_word_file(char_train_path, w2v, batch_size, ll)


