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
        self.init_output = np.random.randn(hd)

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

        outputs, states = [], [self.init_state]
        self.init_weight_output()

        for t in range(self.times):
            ht_1 = self.init_output if t == 0 else outputs[t - 1]
            st_1 = states[t-1]
            xt = np.hstack((x[:, t], ht_1))  # eq.0

            ft = self.act.sigmod(np.dot(self.wf, xt)+self.bf)    # eq.1
            it = self.act.sigmod(np.dot(self.wi, xt)+self.bi)    # eq.2
            ot = self.act.sigmod(np.dot(self.wo, xt)+self.bo)    # eq.3
            at = self.act.tanh(np.dot(self.wa, xt)+self.ba)      # eq.4

            st = ft * st_1 + it * at   # eq.5
            gt = self.act.tanh(st)  # eq.6
            ht = gt*ot   # eq.7

            states.append(st)
            outputs.append(ht)
            self.record_weight_output(ft, it, ot, at, gt)

        return outputs, states

    def backward_propagate(self, x, y, lr=0.002):
        """采用欧几里得距离作为损失函数"""
        out, states = self.forward_propagate(x)

        loss = np.sum((out[-1]-y[:, -1])**2)
        dh = 2*(out[-1]-y[:, -1])
        ds = np.zeros_like(self.init_state)

        swi, swf, swo, swa = np.zeros_like(self.wi), np.zeros_like(self.wf), np.zeros_like(self.wo), np.zeros_like(self.wa)
        sbi, sbf, sbo, sba = np.zeros_like(self.bi), np.zeros_like(self.bf), np.zeros_like(self.bo), np.zeros_like(self.ba)
        for t in range(self.times-1, -1, -1):

            # ds = delta_E/delta_s = delta_E/delta_h*delta_h/delta_s, 根据【eq.7】得到如下结果
            ds += dh * self.zo_list[t] * self.act.tanh_diff(self.zg_list[t])  # 累加
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
            dh = dxi[self.xd:]

            # 求解dw, db
            ht_1 = self.init_output if t == 0 else out[t-1]
            xt = np.hstack((x[:, t], ht_1))

            swi += np.outer(dzi, xt)
            swf += np.outer(dzf, xt)
            swo += np.outer(dzo, xt)
            swa += np.outer(dza, xt)
            sbi += dzi
            sbf += dzf
            sbo += dzo
            sba += dza

            loss += np.sum((out[t] - y[:, t])**2)

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
        tag_dict = {0: 'S', 1: 'B', 2: 'M', 3: 'E'}
        state, out = self.forward_propagate(x)
        predict = []
        for o in out:
            idx = np.argmax(o)
            predict.append(idx)
        predict = map(lambda x: tag_dict.get(x), predict)
        return predict


if __name__ == '__main__':
    from rnn import getrandomdata, test

    lstm = BaseLSTM(10, 10)

    epoches = 50
    smooth_loss = 0
    x, y = getrandomdata(2000)  # 同一批数据训练epoches次
    for ll in range(epoches):
        print('epoch i:', ll)
        # x, y = getrandomdata(2000)  # 所有数据分成epoches次训练
        for i in range(x.shape[0]):
            loss = lstm.backward_propagate(x[i], y[i], lr=0.001)
            if i == 1:
                smooth_loss = loss
            else:
                smooth_loss = smooth_loss * 0.999 + loss * 0.001
        print('loss ----  ', smooth_loss)
        # test(7)








