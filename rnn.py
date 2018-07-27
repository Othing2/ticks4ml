# coding=utf-8

"""

refer: [1] https://github.com/qixianbiao/RNN/blob/aac3746c23a8de9ba8eb13ef189aacb435c7337f/rnn.py
       [2] http://ir.hit.edu.cn/~jguo/docs/notes/bptt.pdf
       [3] https://zybuluo.com/hanbingtao/note/541458
       [4] http://nicodjimenez.github.io/2014/08/08/lstm.html
       [5] https://blog.aidangomez.ca/2016/04/17/Backpropogating-an-LSTM-A-Numerical-Example/
"""

import random
import copy
import numpy as np


class BaseRNN(object):
    def __init__(self, dim_in, dim_hidden, dim_out, if_biase=False):
        self.nn = 3
        self.layers = [dim_in, dim_hidden, dim_out]

        self.times = 0
        # 输入方向权重
        self.w_weights = [np.random.randn(y, x) for x, y in zip(self.layers[:-1], self.layers[1:])]
        # 时间序列方向权重
        self.u_weights = [np.random.rand(y, y) for y in self.layers[1:-1]]
        # 偏置
        self.w_biases = [np.random.randn(y, 1) for y in self.layers[1:]]
        self.u_biases = [np.random.rand(y, 1) for y in self.layers[1:-1]]
        # 时间序列隐藏层的状态值
        self.init_state = np.random.randn(dim_hidden, 1)

        self.activator = self.fun_tanh
        self.activator_div = self.div_tanh

        self.if_biase = if_biase

    def forward_propagate(self, x):
        """ 输入 x: 一个经过标准的序列，而不是序列中的一个元素"""
        self.times = x.shape[1]
        u, wi, wo = self.u_weights[0], self.w_weights[0], self.w_weights[-1]

        output, states = [], []

        for t in range(self.times):
            # 隐藏层
            a = x[:, t].reshape(-1, 1)  # t时刻输入, array切片把列向量转换成行向量了
            pre_state = self.init_state if t == 0 else states[t-1]

            z = np.dot(u, pre_state)+np.dot(wi, a)  # 隐藏层加权输入
            ah = self.activator(z)  # 隐藏层状态值, 激活函数采用tanh()
            states.append(ah)

            # 输出层
            zo = np.dot(wo, ah) # 输出层加权输入
            ao = self.softmax(zo)  # 输出层状态值, 输出采用softmax()
            output.append(ao)

        return output, states

    def backward_propagate(self, x, y, lr=0.002):
        """BPTT 算法: 损失函数采用交叉熵误差函数, 其 delta=out-y """

        u, wi, wo = self.u_weights[0], self.w_weights[0], self.w_weights[-1]
        du, dwi, dwo = np.zeros(u.shape), np.zeros(wi.shape), np.zeros(wo.shape)

        out, states = self.forward_propagate(x)

        yt = y[:, -1].reshape(-1, 1)

        loss = np.sum(-yt*np.log(out[-1]))
        delta_ht = np.dot(wo.transpose(), out[-1]-yt)

        for t in range(self.times-2, -1, -1):
            yt = y[:, t].reshape(-1, 1)
            xt = x[:, t].reshape(-1, 1)
            delta_o = out[t] - yt
            dwo += np.outer(delta_o, states[t])

            delta_ht += np.dot(wo.transpose(), out[t]-yt)  # 注意： delta_h的系数是往后的各时刻的累加, 可以看refer[2]论文(9)等式
            delta_h = delta_ht*self.activator_div(states[t])
            dwi += np.outer(delta_h, xt)
            du += np.outer(delta_h, states[t-1])

            loss += np.sum(-yt*np.log(out[t]))

        self.u_weights[0] -= lr*du/np.sqrt(du*du+0.00000001)
        self.w_weights[0] -= lr*dwi/np.sqrt(dwi*dwi+0.00000001)
        self.w_weights[-1] -= lr*dwo/np.sqrt(dwo*dwo+0.00000001)

        self.init_state -= lr*delta_ht/np.sqrt(delta_ht*delta_ht+0.00000001)  # 更新初始化状态,不知道有什么特殊含义
        return loss

    def fun_tanh(self, weighted_input):
        return np.tanh(weighted_input)

    def div_tanh(self, state):
        return 1 - state*state

    def fun_relu(self, weighted_input):
        """Relu 函数"""
        def op(e): return max(0, e)

        z = copy.deepcopy(weighted_input)
        for e in np.nditer(z, op_flags=['readwrite']):
            e[...] = op(e)
        return z

    def div_relu(self, output):
        """Relu 函数导数"""
        def op(e): return 1 if e > 0 else 0

        o = copy.deepcopy(output)
        for e in np.nditer(o, op_flags=['readwrite']):
            e[...] = op(e)
        return o

    def softmax(self, z):
        z = (np.exp(z) + 0.00000000001) / np.sum(np.exp(z) + 0.00000000001)
        return z

    def sample(self, x):
        h = self.init_state
        u, wi, wo = self.u_weights[0], self.w_weights[0], self.w_weights[-1]
        predict = []
        for i in range(9-1):
            ht = self.activator(np.dot(wi, x) + np.dot(u, h))
            ot = self.softmax(np.dot(wo, ht))
            ynext = np.argmax(ot)
            predict.append(ynext)
            x = np.zeros_like(x)
            x[ynext] = 1
        return predict

    def predict(self, x):
        tag_dict = {0: 'S', 1: 'B', 2: 'M', 3: 'E'}
        out, state = self.forward_propagate(x)
        predict = []
        for o in out:
            idx = np.argmax(o)
            predict.append(idx)
        predict = map(lambda x: tag_dict.get(x), predict)
        return predict


# --------------------------------------------------------------
# create 2000 sequences with 10 number in each sequence
def getrandomdata(nums):
    x = np.zeros([nums, 10, 9], dtype=float)
    y = np.zeros([nums, 10, 9], dtype=float)
    for i in range(nums):
        tmpi = np.random.randint(0, 9)
        for j in range(9):
            if tmpi < 8:
                x[i, tmpi, j], y[i, tmpi+1, j] = 1.0, 1.0
                tmpi = tmpi+1
            else:
                x[i, tmpi, j], y[i, 0, j] = 1.0, 1.0
                tmpi = 0
    return x, y


def test(nums):
    testx = np.zeros([nums, 10], dtype=float)
    for i in range(nums):
        tmpi = np.random.randint(0, 9)
        testx[i, tmpi] = 1
    for i in range(nums):
        print('the given start number:', np.argmax(testx[i]))
        print('the created numbers:   ', model.sample(testx[i].reshape(-1, 1)))


if __name__ == '__main__':
    # x0 = [0, 1, 2, 3, 4, 5, 6, 7, 8]--> y0 = [1, 2, 3, 4, 5, 6, 7, 8, 0],
    # x1 = [5, 6, 7, 8, 0, 1, 2, 3, 4]--> y1 = [6, 7, 8, 0, 1, 2, 3, 4, 5]

    model = BaseRNN(10, 200, 10)

    epoches = 5
    smooth_loss = 0
    x, y = getrandomdata(2000)  # 同一批数据训练epoches次
    for ll in range(epoches):
        print('epoch i:', ll)
        # x, y = getrandomdata(2000)  # 所有数据分成epoches次训练
        for i in range(x.shape[0]):
            loss, state = model.backward_propagate(x[i], y[i], lr=0.001)
            if i == 1:
                smooth_loss = loss
            else:
                smooth_loss = smooth_loss * 0.999 + loss * 0.001
        print('loss ----  ', smooth_loss)
        test(7)
# ----------------------------------------------------------------------------------------


# def load_word_file(path, w2v, bath_size=0, bath_id=0):
#     # [0/S 1/B 2/M 3/E]
#     w_dim = w2v.WordDim
#     max_len = 80
#     with open(path, 'r') as f:
#         word_tags = []
#         if bath_size > 0:
#             lines = f.readlines()[bath_size*bath_id:bath_size*(bath_id+1)]
#         else:
#             lines = f.readlines()
#         for line in lines:
#             ln = map(lambda x: int(x), line.strip().split(' '))
#             word = filter(lambda x: x != 0, ln[:max_len])
#             tag = ln[max_len:max_len+len(word)]
#
#             assert len(word) == len(tag)
#             vect_x = np.zeros([w_dim, len(word)])
#             vect_y = np.zeros([4, len(word)])
#             for i, w in enumerate(word):
#                 v = w2v.GetVectorByIndex(w)
#                 if not v:
#                     print("Not exit", w)
#                     continue
#                 vect_x[:, i] = v
#                 vect_y[tag[i], i] = 1.0
#             word_tags.append((vect_x, vect_y))
#     return word_tags
#
#
# def word_test(path, w2v, ll=0):
#     tag_dict = {0: 'S', 1: 'B', 2: 'M', 3: 'E'}
#     w_dim = w2v.WordDim
#     max_len = 80
#     with open(path, 'r') as f:
#         line = f.readline()
#         if ll:
#             line = f.readlines()[:ll][-1]
#         ln = map(lambda x: int(x), line.strip().split(' '))
#         word = filter(lambda x: x != 0, ln[:max_len])
#         tag = ln[max_len:max_len + len(word)]
#
#         words = [w2v.GetWordByIndex(w) for w in word]
#         tags = ''.join([tag_dict.get(t) for t in tag])
#         lt = len(words)
#         for i, t in enumerate(tags[::-1]):
#             if t == 'S' or t == 'E':
#                 if i == 0:
#                     words.append('')
#                 else:
#                     words.insert(lt-i, '/')
#
#     return u''.join(words)
#
#
# if __name__ == '__main__':
#     """测试中文分词, 效果太差了 ^_^"""
#     from refers.tag_build._config import *
#     from refers.tag_build.w2v import Word2vecVocab
#
#     w2v = Word2vecVocab()
#     w2v.Load(char_vec_path)
#     model = BaseRNN(50, 100, 4)
#
#     ws = word_test(char_train_path, w2v, ll=4)
#     print(ws)
#
#     # test_word = u"人民富裕起来了"  # B E B E B M E
#     # test_vect = np.zeros([w2v.WordDim, len(test_word)])
#     # for i, w in enumerate(test_word):
#     #     v = w2v.GetVector(w.encode("utf8"))
#     #     test_vect[:, i] = v
#     #
#     # epoches = 70
#     # batch_size = 10000
#     # smooth_loss = 0
#     # for ll in range(epoches):
#     #     print('epoch i:', ll)
#     #     train_word = load_word_file(char_train_path, w2v, batch_size, ll)
#     #     for i, (x, y) in enumerate(train_word):
#     #         loss, state = model.backward_propagate(x, y, lr=0.001)
#     #         if i == 1:
#     #             smooth_loss = loss
#     #         else:
#     #             smooth_loss = smooth_loss * 0.999 + loss * 0.001
#     #     print('loss ----  ', smooth_loss)




