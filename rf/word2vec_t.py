# coding=utf-8
"""
# 词向量

"""
import word2vec as w2v
from word2vec import word2vec

# -------------------------------------------------------------
"""
# 字向量
# 设置 _config.py 中相应的文件路径
python prepare_char.py   # 这一步是去除语料库的标注信息，并且将每个字都分开
word2vec -train pre_chars_for_w2v.txt -save-vocab pre_vocab.txt -min-count 3       # 先得到初步词表
python replace_unk.py   # 处理低频词
word2vec -train chars_for_w2v.txt -output vec.txt -size 50 -sample 1e-4 -negative 5 -hs 1 -binary 0 -iter 5  # 训练word2vec 这一步可以得到每个字的字向量
python filter_sentence.py  # 生成分词 训练集 和 测试集

"""
txt_path = "/home/worker/backup/people_2014/pre_chars_for_w2v.txt"
vec_path = "/home/worker/backup/people_2014/pre_vocab.txt"
unk_path = "/home/worker/backup/people_2014/chars_for_w2v.txt"
out_path = "/home/worker/backup/people_2014/char_vec.txt"

# 先得到初步词表
# word2vec(txt_path, vec_path, binary=0, save_vocab=True, min_count=3, verbose=True)

# 训练word2vec 这一步可以得到每个字的字向量
# word2vec(unk_path, out_path, size=50, sample=1e-4, negative=5, hs=1, binary=0, iter_=5, verbose=True)

char_model = w2v.load(out_path)

vec = char_model.get_vector(u"红")

print(vec)




