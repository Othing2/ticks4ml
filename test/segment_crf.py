# -*- coding: utf-8 -*-

import sys
sys.path.append("/home/worker/backup/ticks4ml/")

from keras.models import Model, load_model
import numpy as np
import re
from nlp.segment import SegmentSetting

# 读取字典
vocab = open('../data/msr/msr_training_words.utf8').read().rstrip('\n').split('\n')
vocab = list(''.join(vocab))
stat = {}
for v in vocab:
    stat[v] = stat.get(v, 0) + 1
stat = sorted(stat.items(), key=lambda x: x[1], reverse=True)
vocab = [s[0] for s in stat]
vocabSize = len(vocab) + 1
# 映射
char2id = {c: i + 1 for i, c in enumerate(vocab)}
id2char = {i + 1: c for i, c in enumerate(vocab)}
tags = {'s': 0, 'b': 1, 'm': 2, 'e': 3, 'x': 4}

# 定义参数
embedding_size = 128
maxlen = 32  # 长于32则截断，短于32则填充0
hidden_size = 64
batch_size = 64
epochs = 2

seg = SegmentSetting('bilstm_cnn_crf')
model = seg.load_model('../model/segment/bilstm_cnn_crf2.h5')


def cut_words_crf(data):
    data = re.split('[，。！？、\n]', data)
    sens = []
    Xs = []
    for sentence in data:
        sen = []
        X = []
        sentence = list(sentence)
        for s in sentence:
            s = s.strip()
            if not s == '' and s in char2id:
                sen.append(s)
                X.append(char2id[s])
        if len(X) > maxlen:
            sen = sen[:maxlen]
            X = X[:maxlen]
        else:
            for i in range(maxlen - len(X)):
                X.append(0)

        if len(sen) > 0:
            Xs.append(X)
            sens.append(sen)

    Xs = np.array(Xs)
    ys = model.predict(Xs)
    ys_label = np.argmax(ys, axis=2)

    pred_text_all = ''
    pred_label_all = []
    for text, label in zip(Xs, ys_label):
        pred_text, pred_label = create_pred_text(text, label)
        pred_text_all += pred_text
        pred_label_all.extend(pred_label)

    return pred_text_all, pred_label_all


print(cut_words_crf('中国共产党第十九次全国代表大会，是在全面建成小康社会决胜阶段、中国特色社会主义进入新时代的关键时期召开的一次十分重要的大会。'))

print(cut_words('中国共产党第十九次全国代表大会，是在全面建成小康社会决胜阶段、中国特色社会主义进入新时代的关键时期召开的一次十分重要的大会。'))
print(cut_words('把这本书推荐给，具有一定编程基础，希望了解数据分析、人工智能等知识领域，进一步提升个人技术能力的社会各界人士。'))
print(cut_words('结婚的和尚未结婚的。'))
print(cut_words('管红光出生在江西一个农村，从小立志当科学家。结果成了养猪饲料员。'))

# print(cut_words("""RNN 的 意思 是 ， 为了 预测 最后 的 结果 ， 我 先 用 第一个 词 预测 ， 当然 ， 只 用 第一个 预测 的 预测 结果 肯定 不 精确 ， 我 把 这个 结果 作为 特征 ， 跟 第二词 一起 ， 来 预测 结果 ； 接着 ， 我 用 这个 新 的 预测 结果 结合 第三词 ， 来 作 新 的 预测 ； 然后 重复 这个 过程 。
#
# 结婚 的 和 尚未 结婚 的
#
# 苏剑林 是 科学 空间 的 博主 。
#
# 广东省 云浮市 新兴县
#
# 魏则西 是 一 名 大学生
#
# 这 真是 不堪入目 的 环境
#
# 列夫·托尔斯泰 是 俄罗斯 一 位 著名 的 作家
#
# 保加利亚 首都 索非亚 是 全国 政治 、 经济 、 文化中心 ， 位于 保加利亚 中 西部
#
# 罗斯福 是 第二次世界大战 期间 同 盟国 阵营 的 重要 领导人 之一 。 1941 年 珍珠港 事件发生 后 ， 罗斯 福力 主对 日本 宣战 ， 并 引进 了 价格 管制 和 配给 。 罗斯福 以 租 借 法案 使 美国 转变 为 “ 民主 国家 的 兵工厂 ” ， 使 美国 成为 同 盟国 主要 的 军火 供应商 和 融资 者 ， 也 使得 美国 国内 产业 大幅 扩张 ， 实现 充分 就业 。 二战 后期 同 盟国 逐渐 扭转 形势 后 ， 罗斯福 对 塑造 战后 世界 秩序 发挥 了 关键 作用 ， 其 影响 力 在 雅尔塔 会议 及 联合国 的 成立 中 尤其 明显 。 后来 ， 在 美国 协助 下 ， 盟军 击败 德国 、 意大利 和 日本 。"""))
