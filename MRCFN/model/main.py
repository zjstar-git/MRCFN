# -*- coding: UTF-8 -*-
"""
@author: Jie Zhang
@datetime: 2024/12/14 20:51 星期六
"""
from index import tf_index_combine
import os
from PIL import Image
import numpy as np
from module.fuse_model import create_multi_input_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import scipy.io


def calculate_metrics(testPredict, testY, column_index):
    y_pred = testPredict[:, column_index]
    y_true = testY[:, column_index]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return rmse, mape, mae, r2


def plot_comparison(testPredict, testY, column_index, ax):
    ax.plot(testPredict[:, column_index], label=f'Prediction (Column {column_index+1})', color='blue')
    ax.plot(testY[:, column_index], label=f'Actual (Column {column_index+1})', color='orange')
    ax.set_title(f'Comparison of Column {column_index+1}')
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True)


def create_sequences_t2(data, sequence_length, target_columns):
    X, y, y1= [], [], []
    for i in range(len(data) - sequence_length-1):
        X.append(data.iloc[i:i+sequence_length].values)
        y.append(data.iloc[i+sequence_length][target_columns].values)
        y1.append(data.iloc[i+sequence_length+1][target_columns].values)
    combined_list = [np.concatenate((a, b)) for a, b in zip(y, y1)]
    return X, combined_list


def create_sequences(data, sequence_length, target_columns):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data.iloc[i:i+sequence_length].values)
        y.append(data.iloc[i+sequence_length][target_columns].values)
    return X, y


if __name__ == '__main__':
    index_file_path = '../b.xlsx'
    TF_file_path = '../data_set/40307.xlsx'

    tf_data = tf_index_combine.timeseries_data(index_file_path, TF_file_path)

    tf_data_numeric = tf_data.drop(columns=['data'])


    # 初始化MinMaxScaler
    tf_data_numeric = tf_data_numeric.astype('float64')
    scaler1 = MinMaxScaler()
    tf_data_numeric1 = pd.DataFrame(scaler1.fit_transform(tf_data_numeric), columns=tf_data_numeric.columns)


    # 输入数据归一化消除量纲
    sequence_length = 10
    target_columns = ['max_energy', 'avg_energy', 'event_frequency']

    # 构造输入 X 和目标 y
    # X, y = create_sequences(tf_data_numeric1, sequence_length, target_columns)

    # t+2
    X, y = create_sequences_t2(tf_data_numeric1, sequence_length, target_columns)

    X = np.array(X)

    y = np.array(y)

    # X1, y1 = create_sequences(tf_data_numeric, sequence_length, target_columns)
    # t+2
    X1, y1 = create_sequences_t2(tf_data_numeric, sequence_length, target_columns)

    y1 = np.array(y1)
    # 输出数据归一化消除量纲
    scaler2 = MinMaxScaler(feature_range=(0, 1))
    y1 = scaler2.fit_transform(y1)


    # 3. 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)  # 按时间顺序分割



    # 输出数据形状
    print("训练集 X:", X_train.shape)
    print("训练集 y:", y_train.shape)
    print("测试集 X:", X_test.shape)
    print("测试集 y:", y_test.shape)

    # 定义图片文件夹路径
    # folder_path = '../img'  # 替换为你的文件夹路径
    folder_path = '../index/output'  # 替换为你的文件夹路径

    # 初始化 img_data 列表
    img_data = []

    # 遍历图片文件
    for i in range(1, 377):  # 从001到376
        file_name = f"{i:03}.jpg"  # 格式化为三位数字
        file_path = os.path.join(folder_path, file_name)

        if os.path.exists(file_path):  # 检查文件是否存在
            try:
                # 打开图像
                with Image.open(file_path) as img:
                    # 转换为RGB（防止灰度图等问题）
                    img = img.convert('RGB')
                    # 调整图片尺寸
                    img_array = img.resize((500, 210))
                    # 转换为NumPy数组
                    img_array = np.array(img_array)

                    # 归一化到[0,1]
                    img_array = img_array / 255.0
                    # 添加到img_data
                    img_data.append(img_array)
            except Exception as e:
                print(f"无法处理文件 {file_name}: {e}")
        else:
            print(f"文件 {file_name} 不存在")

    # 转换为NumPy数组
    img_data = np.array(img_data)
    # 切分数据集
    train_img = img_data[:300]
    test_img = img_data[300:375]
    # 模型参数
    image_input_shape = (210, 500, 3)  # 图像大小
    signal_input_shape = (10, 9)  # 信号输入大小

    # 创建模型
    model = create_multi_input_model(image_input_shape, signal_input_shape)

    # 编译模型
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])

    # 打印模型摘要
    model.summary()

    # 训练模型
    history = model.fit([train_img, X_train], y_train, epochs=500, batch_size=32, validation_data=([test_img, X_test], y_test), verbose=2)
    # 模型预测
    # trainPredict = model.predict(trainX)
    testPredict = model.predict([test_img, X_test])
    # 对测试数据进行逆缩放
    testPredict = scaler2.inverse_transform(testPredict)
    testY = scaler2.inverse_transform(y_test)
    # 创建子图
    fig, axs = plt.subplots(6, 1, figsize=(10, 15))
    # 计算所有列的指标并保存到字典
    metrics = {}
    for i in range(6):
        # 计算指标
        rmse, mape, mae, r2 = calculate_metrics(testPredict, testY, i)
        metrics[f'Column_{i + 1}'] = {
            'RMSE': rmse,
            'MAPE': mape,
            'MAE': mae,
            'R2': r2
        }
        plot_comparison(testPredict, testY, i, axs[i])
        # 在标题中添加指标信息
        axs[i].set_title(
            f'Column {i + 1} Comparison\nRMSE: {rmse:.4f}, MAPE: {mape:.2f}%, MAE: {mae:.4f}, R^2: {r2:.4f}')

    plt.tight_layout()
    plt.show()
    print("结束")
    # T+1
    # scipy.io.savemat("../result_data/main/index.mat", metrics)
    # true = {'my_array': testY}
    # scipy.io.savemat("../result_data/main/true.mat", true)
    # predict = {'my_array': testPredict}
    # scipy.io.savemat("../result_data/main/predict.mat", predict)
    # val_loss = {'my_array': history.history['val_loss']}
    # scipy.io.savemat("../result_data/main/val_loss.mat", val_loss)

    # T+2
    scipy.io.savemat("../result_data_2/main/index.mat", metrics)
    true = {'my_array': testY}
    scipy.io.savemat("../result_data_2/main/true.mat", true)
    predict = {'my_array': testPredict}
    scipy.io.savemat("../result_data_2/main/predict.mat", predict)
    # testPredict.to_excel('../result_data/cnn/predict.xlsx')
    val_loss = {'my_array': history.history['val_loss']}
    scipy.io.savemat("../result_data_2/main/val_loss.mat", val_loss)