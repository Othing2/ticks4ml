# -*- coding: UTF-8 -*-

"""
refer: http://ufldl.stanford.edu/wiki/index.php/神经网络
       https://www.jianshu.com/p/679e390f24bb

"""
import random
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


class BaseNN(object):

    def __init__(self, layer_list):
        """
        sample_dt: sample data. [(np.array, label)]
        """
        self.nn = len(layer_list)
        self.layers = layer_list

        # weight
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_list[:-1], layer_list[1:])]
        # bias
        self.biases = [np.random.randn(y, 1) for y in layer_list[1:]]

        self.act = Activiator()

    def forward_output(self, a):
        """只返回最后输出结果, 不保留中间层的输入输出值"""
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            a = self.act.sigmod(z)
        return a

    def forward_propagate(self, a):
        forward_a = [a]  # 每层神经元的输出向量 [Nx1]维, 初始化为网络输入
        forward_z = []   # 每层神经元加权输入向量 [Nx1]维, 输入层没有z
        for b, w in zip(self.biases, self.weights):   # loop: a = f(z)
            z = np.dot(w, a)+b
            forward_z.append(z)
            a = self.act.sigmod(z)
            forward_a.append(a)
        return forward_a, forward_z

    def backward_propagate(self, x, y):
        db = [np.zeros(b.shape) for b in self.biases]
        dw = [np.zeros(w.shape) for w in self.weights]

        # 向前计算每层的输入,输出值
        al, zl = self.forward_propagate(x)
        # 向后计算残差
        dt = (al[-1]-y)*self.act.sigmod_diff(zl[-1])
        db[-1] = dt
        dw[-1] = np.dot(dt, al[-2].transpose())
        for i in range(2, self.nn):        # loop: dt = w*dt*df(z)
            dt = np.dot(self.weights[-i+1].transpose(), dt)*self.act.sigmod_diff(zl[-i])
            db[-i] = dt
            dw[-i] = np.dot(dt, al[-i-1].transpose())
        return db, dw

    def tran_gd(self, alt, eps, sample_dt, test_dt=None):
        """梯度下降算法： 所有样本的训练一次性完成, 效果很差, 收敛太慢"""
        print("开始训练，较耗时，请稍等。。。")
        if test_dt: tn = len(test_dt)

        for j in range(eps):
            # 搅乱训练集，让其排序顺序发生变化
            random.shuffle(sample_dt)

            ndb = [np.zeros(b.shape) for b in self.biases]
            ndw = [np.zeros(w.shape) for w in self.weights]
            for x, y in sample_dt:
                db, dw = self.backward_propagate(x, y)

                ndb = [nb+b for nb, b in zip(ndb, db)]
                ndw = [nw+w for nw, w in zip(ndw, dw)]

            self.weights = [w-(alt/len(sample_dt))*nw for w, nw in zip(self.weights, ndw)]
            self.biases = [b-(alt/len(sample_dt))*nb for b, nb in zip(self.biases, ndb)]

            if test_dt:
                print("Epoch {0}: {1} / {2}".format(j, self.test(test_dt), tn))

    def tran_batch(self, alt, eps, batch_size, sample_dt, test_dt=None):
        """批量梯度下降： 所有样本分批次训练, 每批次训练batch_size个样本"""
        print("开始分批次训练，较耗时，请稍等。。。")

        if test_dt: tn = len(test_dt)

        for j in range(eps):
            # 搅乱训练集，让其排序顺序发生变化
            random.shuffle(sample_dt)
            # 按照小样本数量划分训练集
            batch_list = [sample_dt[k:k+batch_size] for k in range(0, len(sample_dt), batch_size)]
            for batch_sample in batch_list:
                # 根据每个小样本来更新 w 和 b
                self.tran_mini_batch(alt, batch_sample)
            if test_dt:
                print("Epoch {0}: {1} / {2}".format(j, self.test(test_dt), tn))

    def tran_random(self):
        """随机梯度下降算法： """
        raise NotImplementedError

    def tran_mini_batch(self, alt, batch_sample):
        ndb = [np.zeros(b.shape) for b in self.biases]
        ndw = [np.zeros(w.shape) for w in self.weights]
        for x, y in batch_sample:
            # self.forward_propagate(x)
            db, dw = self.backward_propagate(x, y)
            # 加法顺序很重要, 亲测 如果是 b+nb, 则结果会出现很大误差
            ndb = [nb+b for nb, b in zip(ndb, db)]
            ndw = [nw+w for nw, w in zip(ndw, dw)]

        self.weights = [w - (alt / len(batch_sample)) * nw for w, nw in zip(self.weights, ndw)]
        self.biases = [b - (alt / len(batch_sample)) * nb for b, nb in zip(self.biases, ndb)]

    def test(self, test_dt=None):
        res = [(np.argmax(self.forward_output(x)), y) for (x, y) in test_dt]
        return sum(int(x == y) for (x, y) in res)


if __name__ == '__main__':
    """全连接网络 识别 手写数字"""
    from help.BP.mnist_loader import load_data_wrapper
    training_data, validation_data, test_data = load_data_wrapper()
    bn = BaseNN([784, 30, 10])
    # bn.tran(3.0, 30, sample_dt=training_data, test_dt=test_data)
    bn.tran_batch(3.0, 30, 10, sample_dt=training_data, test_dt=test_data)



