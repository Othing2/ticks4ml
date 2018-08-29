from keras.layers import Input, Embedding, LSTM, Dense, concatenate, \
    Conv2D, Flatten, MaxPool2D
from keras.models import Model

"""
多输入多输出网络模型
"""
x_main_data = None
x_aux_data = None
y_labels = None

main_input = Input(shape=(100, ), dtype='int32', name='main_input')
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)
lstm_out = LSTM(32)(x)

aux_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)

aux_input = Input(shape=(5, ), name='aux_input')
x = concatenate([lstm_out, aux_input])
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
main_output = Dense(1, activation='sigmoid', name='main_output')(x)

model = Model(inputs=[main_input, aux_input], outputs=[main_output, aux_output])
# ------------
model.compile(optimizer='rmsprop', loss='binary_crossentropy', loss_weights=[1., 0.2])
model.fit([x_main_data, x_aux_data], [y_labels, y_labels], epochs=50, batch_size=32)
# or ---------
model.compile(optimizer='rmsprop',
              loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
              loss_weights={'main_output': 1., 'aux_output': 0.2})
model.fit(x={'main_input': x_main_data, 'aux_input': x_aux_data},
          y={'main_output': y_labels, 'aux_output': y_labels},
          epochs=50, batch_size=32)
# -----------

"""
共享网络层
例如判断两条博客是否来自同一个人，需要对两条博客使用同一种‘编码器’对博客文本转换成编码，在比较两个编码的相似度, 即一个 2输入１输出的网络
"""
blog_a = Input(shape=(140, 256))
blog_b = Input(shape=(140, 256))

#  实例化一个共享网络层
shared_layer = LSTM(64)
# 当我们重用相同的图层实例多次，图层的权重也会被重用 (它其实就是同一层)
encoded_a = shared_layer(blog_a)
encoded_b = shared_layer(blog_b)

merged_vec = concatenate([encoded_a, encoded_b], axis=-1)
predictions = Dense(1, activation='sigmoid')(merged_vec)

model = Model(inputs=[blog_a, blog_b], outputs=predictions)
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit([x_main_data, x_aux_data], y_labels, epochs=10)

# -------------

"""
共享模型
例如判断两个MNIST数字是否相同
"""
digit_input = Input(shape=(27, 27, 1))
x = Conv2D(64, (3, 3))(digit_input)
x = Conv2D(64, (3, 3))(x)
x = MaxPool2D((2, 2))(x)
out = Flatten()(x)

shared_layer = Model(digit_input, out)

digit_a = Input(shape=(27, 27, 1))
digit_b = Input(shape=(27, 27, 1))

out_a = shared_layer(digit_a)
out_b = shared_layer(digit_b)

concatenated = concatenate([out_a, out_b])
out = Dense(1, activation='sigmoid')(concatenated)

classification_model = Model([digit_a, digit_b], out)




