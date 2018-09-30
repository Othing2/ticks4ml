import keras.backend as K
from keras.engine import Layer, InputSpec


class ChainCRF(Layer):
    """
    ref: https://blog.csdn.net/JackyTintin/article/details/79261981
    """
    def __init__(self, **kwargs):
        super(ChainCRF, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        n_classes = input_shape[2]
        n_steps = input_shape[1]
        assert n_classes >= 2
        assert n_steps is None or n_steps >= 2

        # self.input_spec = [InputSpec(dtype=K.floatx(), shape=(None, n_steps, n_classes))]

        self.T = self.add_weight((n_classes, n_classes),
                                 initializer=self.init,
                                 name='{}_U'.format(self.name),
                                 regularizer=self.T_regularizer,
                                 constraint=self.T_constraint)

        self.b_start = self.add_weight((n_classes,),
                                       initializer='zero',
                                       name='{}_b_start'.format(self.name),
                                       regularizer=self.b_start_regularizer,
                                       constraint=self.b_start_constraint)

        self.b_end = self.add_weight((n_classes,),
                                     initializer='zero',
                                     name='{}_b_end'.format(self.name),
                                     regularizer=self.b_end_regularizer,
                                     constraint=self.b_end_constraint)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        self.built = True




        self.built = True

    def call(self, inputs, **kwargs):
        pass

    def compute_output_shape(self, input_shape):
        return input_shape

    def viterbi_decode(self):
        pass

    def _forward(self, log_prob, tags):
        """
        crf的前向计算
        :param log_prob: 批量序列样本的发射概率
        :param tags: 批量序列样本标签矩阵
        :return:
        """
        batch_size, seq_len, voc = log_prob.size()
        log_alpha = K.zeros((batch_size, voc))

        if self.b_start is not None:
            log_alpha = log_alpha + K.expand_dims(self.b_start, axis=0)
        log_alpha = log_alpha + log_prob[:, 0, :]

        for t in range(1, seq_len):
            trans = K.expand_dims(self.T, axis=0)
            emit = K.expand_dims(log_prob[:, t, :], axis=1)
            log_alpha_tm1 = K.expand_dims(log_alpha, axis=2)
            log_alpha = K.logsumexp(trans + emit + log_alpha_tm1, axis=1)

        if self.b_end is not None:
            log_Z = K.logsumexp(log_alpha+K.expand_dims(self.b_end, axis=0), axis=1)  # 归一化因子
        else:
            log_Z = K.logsumexp(log_alpha, axis=1)

        # score for y
        labels_l = tags[:, :-1]
        expanded_size = labels_l.size() + (voc,)
        labels_l = K.expand_dims(labels_l, axis=-1)

        labels_r = tags[:, 1:]
        labels_r = K.expand_dims(labels_r, axis=-1)

        P_row = K.expand_dims(self.T, axis=0).gather(1, labels_l)
        y_transmit_score = P_row.gather(2, labels_r).squeeze(-1)
        y_emit_score = log_prob.gather(2, K.expand_dims(tags, axis=2)).squeeze(-1)

        log_M = K.sum(y_emit_score, axis=1) + K.sum(y_transmit_score, axis=1)

        if self.b_start is not None:
            log_M = log_M + self.b_start.gather(0, tags[:, 0])

        if self.b_end is not None:
            log_M = log_M + self.b_end.gather(0, tags[:, -1])

        # negative likelihood
        nll = log_Z - log_M
        nll = nll.sum(0).view(1)

        if self.size_average:
            nll.div_(batch_size)

        return nll













