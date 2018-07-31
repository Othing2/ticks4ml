# coding=utf-8

"""

refer: https://zybuluo.com/hanbingtao/note/581764
       https://blog.aidangomez.ca/2016/04/17/Backpropogating-an-LSTM-A-Numerical-Example/
       https://github.com/nicodjimenez/lstm/blob/master/lstm.py
       http://arunmallya.github.io/writeups/nn/lstm/index.html#/
"""

import random
import copy
import numpy as np


class Activiator(object):

    @classmethod
    def sigmod(cls, x):
        return 1./(1+np.exp(-x))

    @classmethod
    def sigmod_diff(cls, y):
        return y*(1-y)

    @classmethod
    def tanh(cls, x):
        return np.tanh(x)

    @classmethod
    def tanh_diff(cls, y):
        return 1. - y**2

    @classmethod
    def softmax(cls, z):
        z = (np.exp(z) + 0.00000000001) / np.sum(np.exp(z) + 0.00000000001)
        return z


class BaseLSTM(object):

    def __init__(self, xd, hd):
        """

        :param xd: 输入x的维度
        :param hd: 输出h的维度
        """
        self.xd = xd
        self.hd = hd
        concat_dim = xd+hd
        self.concat_dim = concat_dim

        self.wf = np.random.randn(hd, concat_dim)
        self.wi = np.random.randn(hd, concat_dim)
        self.wo = np.random.randn(hd, concat_dim)
        self.wa = np.random.randn(hd, concat_dim)

        self.bf = np.random.randn(hd)
        self.bi = np.random.randn(hd)
        self.bo = np.random.randn(hd)
        self.ba = np.random.randn(hd)

        self.init_state = np.random.randn(hd)
        self.init_hyper = np.random.randn(hd)

        self.act = Activiator()
        self.times = 0

    def init_weight_output(self):
        # 加权输出
        self.zf_list = []
        self.zi_list = []
        self.zo_list = []
        self.za_list = []
        self.zg_list = []

    def record_weight_output(self, ft, it, ot, at, gt):
        self.zf_list.append(ft)
        self.zi_list.append(it)
        self.zo_list.append(ot)
        self.za_list.append(at)
        self.zg_list.append(gt)

    def forward_propagate(self, x):
        """ 输入 x: 一个经过标准的序列，而不是序列中的一个元素"""
        self.times = x.shape[1]

        hd_out, states = [self.init_hyper], [self.init_state]
        self.init_weight_output()

        for t in range(self.times):
            ht_1 = hd_out[t]
            st_1 = states[t]
            xt = np.hstack((x[:, t], ht_1))  # eq.0

            ft = self.act.sigmod(np.dot(self.wf, xt)+self.bf)    # eq.1
            it = self.act.sigmod(np.dot(self.wi, xt)+self.bi)    # eq.2
            ot = self.act.sigmod(np.dot(self.wo, xt)+self.bo)    # eq.3
            at = self.act.tanh(np.dot(self.wa, xt)+self.ba)      # eq.4

            st = ft * st_1 + it * at   # eq.5
            gt = self.act.tanh(st)  # eq.6
            ht = gt*ot   # eq.7

            states.append(st)
            hd_out.append(ht)

            self.record_weight_output(ft, it, ot, at, gt)

        return hd_out, states

    def backward_propagate(self, x, y, lr=0.002):
        """采用欧几里得距离作为损失函数"""
        hd_out, states = self.forward_propagate(x)

        loss = np.sum((hd_out[-1]-y[:, -1])**2)
        dh = (hd_out[-1]-y[:, -1])
        ds = np.zeros_like(self.init_state)

        swi, swf, swo, swa = np.zeros_like(self.wi), np.zeros_like(self.wf), np.zeros_like(self.wo), np.zeros_like(self.wa)
        sbi, sbf, sbo, sba = np.zeros_like(self.bi), np.zeros_like(self.bf), np.zeros_like(self.bo), np.zeros_like(self.ba)
        for t in range(self.times-1, -1, -1):
            # ds = delta_E/delta_s = delta_E/delta_h*delta_h/delta_s, 根据【eq.7】得到如下结果
            ds += dh * self.zo_list[t] * self.act.tanh_diff(self.zg_list[t])  # 累加状态误差
            # do = delta_E/delta_o = dh * delta_h/delta_o, 根据【eq.7】得到如下结果
            do = dh * self.zg_list[t]

            # 根据【eq.5】和[ds]可以求出如下结果
            di = self.za_list[t] * ds
            da = self.zi_list[t] * ds
            df = states[t] * ds
            ds_1 = self.zf_list[t] * ds

            # 根据【eq.1~eq.4】可求出w的加权输入的导数, 如下结果: zi=wi*xt; dzi=delta_E/delta_zi, 因此, dwi=delta_E/delta_wi=dzi*xt如下同理
            dzi = di * self.act.sigmod_diff(self.zi_list[t])
            dzf = df * self.act.sigmod_diff(self.zf_list[t])
            dzo = do * self.act.sigmod_diff(self.zo_list[t])
            dza = da * self.act.tanh_diff(self.za_list[t])

            # 更新dh, 参照[http://arunmallya.github.io/writeups/nn/lstm/index.html#/11]
            dz = np.hstack((dzi, dzf, dzo, dza))
            w = np.vstack((self.wi, self.wf, self.wo, self.wa))
            # 因为 z=w*x -> dx= delta_E/delta_x= dz*w
            dxi = np.dot(w.T, dz)
            _dh = dxi[self.xd:]
            dh = _dh + (hd_out[t]-y[:, t-1])  # 累加输出误差

            # 求解dw, db
            xt = np.hstack((x[:, t], hd_out[t]))

            swi += np.outer(dzi, xt)
            swf += np.outer(dzf, xt)
            swo += np.outer(dzo, xt)
            swa += np.outer(dza, xt)
            sbi += dzi
            sbf += dzf
            sbo += dzo
            sba += dza

            loss += np.sum((hd_out[t][0] - y[:, t])**2)

        self.wi -= lr*swi/np.sqrt(swi*swi+0.00000001)
        self.wf -= lr*swf/np.sqrt(swf*swf+0.00000001)
        self.wo -= lr*swo/np.sqrt(swo*swo+0.00000001)
        self.wa -= lr*swa/np.sqrt(swa*swa+0.00000001)
        self.bi -= lr*sbi/np.sqrt(sbi*sbi+0.00000001)
        self.bf -= lr*sbf/np.sqrt(sbf*sbf+0.00000001)
        self.bo -= lr*sbo/np.sqrt(sbo*sbo+0.00000001)
        self.ba -= lr*sba/np.sqrt(sba*sba+0.00000001)

        self.init_state -= lr*ds/np.sqrt(ds*ds+0.00000001)

        return loss

    def prodict(self, x):
        self.times = x.shape[1]

        outputs = []

        for t in range(self.times):
            ht_1 = self.init_hyper if t == 0 else outputs[t - 1]
            st_1 = self.init_state if t == 0 else st
            xt = np.hstack((x[:, t], ht_1))  # eq.0

            ft = self.act.sigmod(np.dot(self.wf, xt) + self.bf)  # eq.1
            it = self.act.sigmod(np.dot(self.wi, xt) + self.bi)  # eq.2
            ot = self.act.sigmod(np.dot(self.wo, xt) + self.bo)  # eq.3
            at = self.act.tanh(np.dot(self.wa, xt) + self.ba)  # eq.4

            st = ft * st_1 + it * at  # eq.5
            gt = self.act.tanh(st)  # eq.6
            ht = gt * ot  # eq.7

            outputs.append(ht)

        return outputs


if __name__ == '__main__':
    """测试中文分词, 效果太差了 ^_^"""
    from rnn import word_test, load_word_file
    from refers.tag_build._config import *
    from refers.tag_build.w2v import Word2vecVocab

    w2v = Word2vecVocab()
    w2v.Load(char_vec_path)
    model = BaseLSTM(50, 4)

    ws = word_test(char_train_path, w2v, ll=4)
    print(ws)

    test_word = u"人民富裕起来了"  # B E B E B M E
    test_vect = np.zeros([w2v.WordDim, len(test_word)])
    for i, w in enumerate(test_word):
        v = w2v.GetVector(w.encode("utf8"))
        test_vect[:, i] = v

    epoches = 70
    batch_size = 10000
    smooth_loss = 0
    for ll in range(epoches):
        print('epoch i:', ll)
        train_word = load_word_file(char_train_path, w2v, batch_size, ll)
        for i, (x, y) in enumerate(train_word):
            loss, state = model.backward_propagate(x, y, lr=0.001)
            if i == 1:
                smooth_loss = loss
            else:
                smooth_loss = smooth_loss * 0.999 + loss * 0.001
        print('loss ----  ', smooth_loss)








