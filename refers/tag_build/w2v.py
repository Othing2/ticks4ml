# coding=utf-8

import os
import re
import random
import math
from collections import defaultdict, namedtuple


class Word2vecVocab(object):
    IdxVec = namedtuple('Idx_Vec', ['idx', 'vec'])
    WdVec = namedtuple('Wd_Vec', ['wd', 'vec'])

    def __init__(self):
        self.vocab = defaultdict(Word2vecVocab.IdxVec)
        self.idx_vocab = defaultdict(Word2vecVocab.WdVec)
        self.w_dim = 0
        self.n_word = 0
        self.avg_value = []  # 均值
        self.std_value = []  # 方差

    @property
    def WordDim(self):
        return self.w_dim

    def Load(self, vobPath):
        """加载字向量文件,分析每行"""
        if not os.path.exists(vobPath):
            raise Exception("%s path do not exist" % vobPath)
        with open(vobPath, 'r') as fp:
            head_line = re.split(' ', fp.next().strip())
            self.w_dim = int(head_line[1])
            self.avg_value = [0.0] * self.w_dim
            self.std_value = [0.0] * self.w_dim
            secondToken = ''
            for line in fp:
                word_vec = re.split(' ', line.strip())
                word = word_vec[0]
                vec = [float(i) for i in word_vec[1:]]
                if len(vec) != self.w_dim:
                    print("字 ‘%s’ 维度(%d)不符(应该是%s)" % (word, len(word_vec)-1, self.w_dim))
                if self.vocab.get(word, None):
                    print("字 '%s' 重复" % word)
                if self.n_word == 0 and word != "</s>":
                    print("第一个词应该是 '</s>' ")
                if self.n_word == 1:
                    secondToken = word
                if word == "<UNK>":
                    self.vocab[word] = Word2vecVocab.IdxVec(1, vec)
                    self.idx_vocab[1] = Word2vecVocab.WdVec(word, vec)
                    iv = self.vocab[secondToken]
                    self.vocab[secondToken] = Word2vecVocab.IdxVec(self.n_word, iv.vec)
                    self.idx_vocab[self.n_word] = Word2vecVocab.WdVec(word, iv.vec)

                self.vocab[word] = Word2vecVocab.IdxVec(self.n_word, vec)
                self.idx_vocab[self.n_word] = Word2vecVocab.WdVec(word, vec)
                self.avg_value = [a+v for a, v in zip(self.avg_value, vec)]
                self.std_value = [a+(v*v) for a, v in zip(self.std_value, vec)]
                self.n_word += 1
        if self.n_word > 0:  # 求均值,方差
            self.avg_value = [a/(self.n_word+1) for a in self.avg_value]
            self.std_value = [math.sqrt(std/(self.n_word+1)-avg*avg) for std, avg in zip(self.std_value, self.avg_value)]

    def GetWordIndex(self, word):
        wv = self.vocab.get(word, None)
        if not wv:
            return 1
        return wv.idx

    def GetWordByIndex(self, idx):
        wv = self.idx_vocab.get(idx, None)
        if not wv:
            return ''
        return wv.wd.decode('utf8')

    def GetVectorByIndex(self, idx, opt=1):
        wv = self.idx_vocab.get(idx, None)
        if not wv:
            if opt == 0: return []  # USE_BLANK
            elif opt == 1: return self.vocab["</s>"].vec  # USE_OOV
            elif opt == 2:  # USE_RANDOM
                rd = random.random()
                random.seed(rd)
                oov_feature = [float(random.normalvariate(self.avg_value[i], self.std_value[i])) for i in range(self.w_dim)]
                return oov_feature
        return wv.vec

    def GetVector(self, word, opt=1):
        wv = self.vocab.get(word, None)
        if not wv:
            if opt == 0: return []  # USE_BLANK
            elif opt == 1: return self.vocab["</s>"].vec   # USE_OOV
            elif opt == 2:  # USE_RANDOM
                rd = random.random()
                random.seed(rd)
                oov_feature = [float(random.normalvariate(self.avg_value[i], self.std_value[i]))
                                    for i in range(self.w_dim)]
                return oov_feature
        return wv.vec

    def GetTotalWord(self):
        return self.n_word+1

    def GetStat(self):
        return {"均值向量": self.avg_value, "方差向量": self.std_value}


if __name__ == '__main__':
    from _config import char_vec_path
    vob_path = char_vec_path
    vob = Word2vecVocab()
    vob.Load(vob_path)
    ii = vob.GetWordIndex(u"红".encode("utf8"))
    i2 = vob.GetVectorByIndex(509)
    wd = vob.GetWordByIndex(509)
    print(ii, wd, i2)

