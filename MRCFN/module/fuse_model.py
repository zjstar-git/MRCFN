# -*- coding: UTF-8 -*-
"""
@author: Jie Zhang
@datetime: 2025/01/05 16:32 星期日
"""
import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input, Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, Flatten, Dense,
    Dropout, Embedding, LSTM, BatchNormalization, GlobalAveragePooling2D,
    GlobalAveragePooling1D, Lambda, multiply, Concatenate, Reshape, Permute, Dot
)


# 外积计算
def outer_product(tensors):
    a, b = tensors  # 拆分输入
    a = tf.expand_dims(a, axis=2)  # 将 a 的形状变为 (None, 46, 1)
    b = tf.expand_dims(b, axis=1)  # 将 b 的形状变为 (None, 1, 46)
    return tf.matmul(a, b)


# 定义图像输入的分支
def create_image_branch(input_shape):
    # image_input = Input(shape=input_shape, name="image_input")

    # 多尺度卷积
    conv_1 = Conv2D(32, (3, 3), strides=3, activation="relu", padding="same")(input_shape)
    conv_2 = Conv2D(32, (6, 6), strides=3, activation="relu", padding="same")(input_shape)
    conv_3 = Conv2D(32, (9, 9), strides=3, activation="relu", padding="same")(input_shape)

    # 合并通道轴
    con = Concatenate(axis=-1)([conv_1, conv_2, conv_3])
    image_output = MaxPooling2D(pool_size=(2, 2), strides=2)(con)  # 下采样
    return image_output


# 定义信号输入的分支
def create_sign_branch(input_shape):
    # sign_input = Input(shape=input_shape, name="sign_input")
    sign_input = Permute((2, 1))(input_shape)
    # 一维卷积网络
    conv1d_1 = Conv1D(64, kernel_size=3, strides=1, activation="relu", padding="same")(sign_input)
    bn_1 = BatchNormalization()(conv1d_1)  # 批归一化
    return bn_1


# 空间注意力模块
def spatial_attention(feature_map):
    avg_pool = tf.reduce_mean(feature_map, axis=-1, keepdims=True)

    max_pool = tf.reduce_max(feature_map, axis=-1, keepdims=True)
    concat = Concatenate(axis=-1)([avg_pool, max_pool])
    attention = Conv2D(1, (1, 1), activation="relu", padding="same")(concat)
    attention = BatchNormalization()(attention)
    return multiply([feature_map, attention])


# 通道注意力模块
def channel_attention(feature_map):
    # sign_input = Permute((2, 1))(feature_map)
    conv_1 = Conv1D(64, kernel_size=1, strides=1, activation="relu", padding="same")(feature_map)
    bn_1 = BatchNormalization()(conv_1)
    # bn_1 = Reshape((bn_1.shape[1],))(bn_1)
    avg_pool = GlobalAveragePooling1D()(bn_1)
    # max_pool = tf.reduce_max(feature_map, axis=[1, 2])
    attention = Dense(feature_map.shape[-1], activation="sigmoid")(avg_pool)

    return multiply([feature_map, attention])


# 构建多输入模型
def create_multi_input_model(image_input_shape, signal_input_shape):
    # 图像分支
    image_input = Input(shape=image_input_shape, name="image_input")
    image_output = create_image_branch(image_input)

    # 信号分支
    sign_input = Input(shape=signal_input_shape, name="sign_input")
    bn_1 = create_sign_branch(sign_input)

    # 图像特征处理
    image_attention = spatial_attention(image_output)
    x_1 = Lambda(lambda x: x[0] + x[1])([image_output, image_attention])  # 残差相加
    x_1 = Conv2D(1, (1, 1), activation="relu")(x_1)  # 去掉最后一个维度
    x_1 = Reshape((x_1.shape[1], x_1.shape[2]))(x_1)
    x_1 = Lambda(lambda x: tf.reduce_mean(x, axis=2))(x_1)  # 平均操作
    x_2 = Conv1D(1, kernel_size=1, activation="relu")(bn_1)
    x_2 = BatchNormalization()(x_2)
    x_2 = Reshape((x_2.shape[1], ))(x_2)
    x_3 = Concatenate()([x_1, x_2])

    # 信号特征处理
    signal_attention = channel_attention(bn_1)
    s_1 = Lambda(lambda x: x[0] + x[1])([bn_1, signal_attention])  # 残差相加
    s_2 = Conv2D(64, (1, 1), activation="relu")(image_output)
    s_2 = Lambda(lambda x: tf.reduce_mean(x, axis=2))(s_2)
    s_3 = Concatenate(axis=1)([s_1, s_2])
    s_3 = Permute((2, 1))(s_3)
    s_3 = GlobalAveragePooling1D()(s_3)
    x_fuse = Lambda(lambda x: outer_product(x))([x_3, s_3])
    x_fuse = Flatten()(x_fuse)
    # x_fuse = Dot(axes=1)([x_3, s_3])
    # 特征融合
    # x_fuse = Lambda(lambda x: tf.einsum('i,j->ij', x[0], x[1]))([x_3, s_3])  # 外积操作
    dense_out = Dense(128, activation="relu")(x_fuse)
    outputs = Dense(6, activation="sigmoid", name="output_layer")(dense_out)

    # 构建模型
    model = Model(inputs=[image_input, sign_input], outputs=outputs)
    return model


if __name__ == '__main__':

    # 模型参数
    image_input_shape = (220, 500, 3)  # 图像大小
    signal_input_shape = (10, 9)  # 信号输入大小

    # 创建模型
    model = create_multi_input_model(image_input_shape, signal_input_shape)

    # 编译模型
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # 打印模型摘要
    model.summary()
