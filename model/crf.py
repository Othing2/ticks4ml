from keras.layers import Input, Layer, Embedding, LSTM
from keras.backend import tensorflow_backend as tf


class ChainCRF(Layer):
    def __init__(self, **kwargs):
        super(ChainCRF, self).__init__(**kwargs)

    def build(self, input_shape):
        self.trans_prob = self.add_weight()
        self.state_prob = self.add_weight()
        self.built = True

    def call(self, inputs, **kwargs):
        pass

    def compute_output_shape(self, input_shape):
        return input_shape

    def viterbi_decode(self):
        pass

    def _forward(self, ts, ss):
        """
        crf的前向计算
        :param ts:  转移概率
        :param ss: 发射函数
        :return:
        """














