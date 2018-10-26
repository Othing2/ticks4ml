# -*- coding: utf-8 -*-
# https://juejin.im/post/5ba2ff3a5188255c672ea8e5
from keras.layers import Input, Dense, Embedding, LSTM, Dropout, \
    TimeDistributed, Bidirectional, Masking, ZeroPadding1D, Conv1D, concatenate
from keras.models import Model, load_model
from keras.utils import np_utils
from keras_contrib.layers.crf import CRF
from keras_contrib.utils import save_load_utils
import numpy as np
import re


def bilstm_build(maxlen, vocab_size, emb_size, lstm_unit, tag_cnt=5, drop_rate=0.6):
    """Ref: https://juejin.im/post/5ba2ff3a5188255c672ea8e5"""
    # BiLSTM
    X = Input(shape=(maxlen,), dtype='int32')
    embedding = Embedding(input_dim=vocab_size, output_dim=emb_size, input_length=maxlen, mask_zero=True)(X)
    blstm = Bidirectional(LSTM(lstm_unit, return_sequences=True), merge_mode='concat')(embedding)
    blstm = Dropout(drop_rate)(blstm)
    blstm = Bidirectional(LSTM(lstm_unit, return_sequences=True), merge_mode='concat')(blstm)
    blstm = Dropout(drop_rate)(blstm)
    output = TimeDistributed(Dense(tag_cnt, activation='softmax'))(blstm)

    model = Model(X, output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def bilstm_crf_build(maxlen, vocab_size, emb_size, lstm_unit, tag_cnt=5, drop_rate=0.6):
    # BiLSTM
    X = Input(shape=(maxlen,), dtype='int32')
    embedding = Embedding(input_dim=vocab_size, output_dim=emb_size, input_length=maxlen, mask_zero=True)(X)
    blstm = Bidirectional(LSTM(lstm_unit, return_sequences=True), merge_mode='concat')(embedding)
    blstm = Dropout(drop_rate)(blstm)
    blstm = Bidirectional(LSTM(lstm_unit, return_sequences=True), merge_mode='concat')(blstm)
    blstm = Dropout(drop_rate)(blstm)
    output = TimeDistributed(Dense(tag_cnt, activation='softmax'))(blstm)

    # CRF
    crf = CRF(tag_cnt, sparse_target=True)
    crf_output = crf(output)

    model = Model(inputs=X, outputs=crf_output)
    model.compile(optimizer='adam', loss=crf.loss_function, metrics=[crf.accuracy])

    return model


def bilstm_cnn_crf_build(maxlen, vocab_size, emb_size, lstm_unit, tag_cnt=5, drop_rate=0.1):
    """Ref: https://github.com/shen1994/chinese_bilstm_cnn_crf"""
    # BiLSTM
    X = Input(shape=(maxlen,), dtype='int32')
    embedding = Embedding(input_dim=vocab_size, output_dim=emb_size, input_length=maxlen)(X)
    blstm = Bidirectional(LSTM(lstm_unit, return_sequences=True))(embedding)
    blstm = Dropout(drop_rate)(blstm)
    blstm = Bidirectional(LSTM(lstm_unit, return_sequences=True))(blstm)
    blstm = Dropout(drop_rate)(blstm)
    bilstm = TimeDistributed(Dense(emb_size))(blstm)

    # CNN
    half_window_size = 2
    filter_kernel_number = 64
    padding_layer = ZeroPadding1D(padding=half_window_size)(embedding)
    conv = Conv1D(nb_filter=filter_kernel_number, filter_length=2*half_window_size+1, padding="valid")(padding_layer)
    conv = Dropout(drop_rate)(conv)
    conv = TimeDistributed(Dense(filter_kernel_number))(conv)

    # merge
    rnn_cnn_merge = concatenate([bilstm, conv], axis=2)
    output = TimeDistributed(Dense(tag_cnt))(rnn_cnn_merge)

    # CRF
    crf = CRF(tag_cnt, sparse_target=True)
    crf_output = crf(output)

    model = Model(inputs=X, outputs=crf_output)
    model.compile(optimizer='adam', loss=crf.loss_function, metrics=[crf.accuracy])

    return model


class SegmentSetting:

    def __init__(self, model_name):
        vocab = self.vocab_data()
        # 5167 个字
        self.vocabSize = len(vocab) + 1
        # 映射
        self.char2id = {c: i + 1 for i, c in enumerate(vocab)}
        self.id2char = {i + 1: c for i, c in enumerate(vocab)}
        self.tags = {'s': 0, 'b': 1, 'm': 2, 'e': 3, 'x': 4}
        # 定义参数
        self.embedding_size = 128
        self.maxlen = 32  # 长于32则截断，短于32则填充0
        self.hidden_size = 64
        self.batch_size = 64
        self.epochs = 2

        self.models = dict(
            bilstm=bilstm_build(self.maxlen, self.vocabSize, self.embedding_size, self.hidden_size, drop_rate=0.6),
            bilstm_crf=bilstm_crf_build(self.maxlen, self.vocabSize, self.embedding_size, self.hidden_size, drop_rate=0.6),
            bilstm_cnn_crf=bilstm_cnn_crf_build(self.maxlen, self.vocabSize, self.embedding_size, self.hidden_size, drop_rate=0.15),
        )

        self.model = self.models[model_name]

        self.bilstm_model=bilstm_build(self.maxlen, self.vocabSize, self.embedding_size, self.hidden_size, drop_rate=0.6),
        self.bilstm_crf_model=bilstm_crf_build(self.maxlen, self.vocabSize, self.embedding_size, self.hidden_size, drop_rate=0.6),
        self.bilstm_cnn_crf_model=bilstm_cnn_crf_build(self.maxlen, self.vocabSize, self.embedding_size, self.hidden_size, drop_rate=0.15),

    def load_model(self, h5_path):
        self.model.load_weights(h5_path)
        return self.model

    def train_and_save(self, file_name):
        x_train, y_train = self.load_data('../data/msr/msr_training.utf8')
        x_test, y_test = self.load_data('../data/msr/msr_test_gold.utf8')
        self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs)
        self.model.save('../model/segment/' + file_name)
        print('Train: ', self.model.evaluate(x_train, y_train, batch_size=self.batch_size))
        print('Test: ', self.model.evaluate(x_test, y_test, batch_size=self.batch_size))

    def load_data(self, path):
        data = open(path).read().rstrip('\n')
        # 按标点符号和换行符分隔
        data = re.split('[，。！？、\n]', data)
        print('共有数据 %d 条' % len(data))
        print('平均长度：', np.mean([len(d.replace(' ', '')) for d in data]))

        # 准备数据
        X_data = []
        y_data = []

        for sentence in data:
            sentence = sentence.split(' ')
            X = []
            y = []

            try:
                for s in sentence:
                    s = s.strip()
                    # 跳过空字符
                    if len(s) == 0:
                        continue
                    # s
                    elif len(s) == 1:
                        X.append(self.char2id[s])
                        y.append(self.tags['s'])
                    elif len(s) > 1:
                        # b
                        X.append(self.char2id[s[0]])
                        y.append(self.tags['b'])
                        # m
                        for i in range(1, len(s) - 1):
                            X.append(self.char2id[s[i]])
                            y.append(self.tags['m'])
                        # e
                        X.append(self.char2id[s[-1]])
                        y.append(self.tags['e'])

                # 统一长度
                if len(X) > self.maxlen:
                    X = X[:self.maxlen]
                    y = y[:self.maxlen]
                else:
                    for i in range(self.maxlen - len(X)):
                        X.append(0)
                        y.append(self.tags['x'])
            except:
                continue
            else:
                if len(X) > 0:
                    X_data.append(X)
                    y_data.append(y)

        X_data = np.array(X_data)
        y_data = np_utils.to_categorical(y_data, 5)

        return X_data, y_data

    @classmethod
    def vocab_data(cls):
        vocab = open('../data/msr/msr_training_words.utf8').read().rstrip('\n').split('\n')
        vocab = list(''.join(vocab))
        stat = {}
        for v in vocab:
            stat[v] = stat.get(v, 0) + 1
        stat = sorted(stat.items(), key=lambda x: x[1], reverse=True)
        vocab = [s[0] for s in stat]
        return vocab


if __name__ == '__main__':

    model = SegmentSetting('bilstm')

    model.train_and_save('bilstm.h5')




