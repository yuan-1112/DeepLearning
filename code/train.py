from __future__ import print_function
import os
import sys
import gzip
import math
import paddle
import paddle.fluid as fluid
import numpy as np
import io
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc, precision_recall_curve
import seaborn as sns

paddle.enable_static()

# 加载词典数据
with io.open("word_dict.txt", "r", encoding="utf-8") as input:
    word_dict = eval(input.read())
    print(len(word_dict))
BATCH_SIZE = 16  # 增大批量大小

# 训练集生成器
def train_generator():
    with io.open("train_data.txt", "r", encoding="utf-8") as output:
        train_data = eval(output.read())
        print(len(train_data))
    def reader():
        for word_vector, label in train_data:
            yield word_vector, label
    return reader

# 测试集生成器
def test_generator():
    with io.open("train_data.txt", "r", encoding="utf-8") as output:
        test_data = eval(output.read())
    def reader():
        for word_vector, label in test_data:
            yield word_vector, label
    return reader

# 数据分Batch处理, 并打乱减少相关性束缚,放数据进显卡
train_reader = paddle.batch(
    paddle.reader.shuffle(
        train_generator(),
    buf_size=51200),
    batch_size=BATCH_SIZE)
test_reader = paddle.batch(
    test_generator(),
    batch_size=BATCH_SIZE)
# for data in test_reader():
#             print(data)
#             print(len(data))
dict_dim = len(word_dict)

# 增加Dropout层和改进模型结构
def lstm_net(data, label, dict_dim, emb_dim=128, hid_dim=128, hid_dim2=96, class_dim=2, emb_lr=30.0):
    emb = fluid.layers.embedding(
        input=data,
        size=[dict_dim, emb_dim],
        param_attr=fluid.ParamAttr(learning_rate=emb_lr))

    fc0 = fluid.layers.fc(input=emb, size=hid_dim * 4)
    lstm_h, c = fluid.layers.dynamic_lstm(
        input=fc0, size=hid_dim * 4, is_reverse=False)

    lstm_max = fluid.layers.sequence_pool(input=lstm_h, pool_type='max')
    lstm_max_tanh = fluid.layers.tanh(lstm_max)
    fc1 = fluid.layers.fc(input=lstm_max_tanh, size=hid_dim2, act='tanh')

    # 添加Dropout层
    dropout = fluid.layers.dropout(x=fc1, dropout_prob=0.5)

    prediction = fluid.layers.fc(input=dropout, size=class_dim, act='softmax')
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)
    return avg_cost, acc, prediction
# 栈式双向LSTM网络结构
def stacked_lstm_net(data,label, input_dim, class_dim=2, emb_dim=128, hid_dim=512, stacked_num=3):
    # 由于设置奇数层正向, 偶数层反向, 最后一层LSTM网络必定正向, 所以栈数必定为奇数
    assert stacked_num % 2 == 1

    emb = fluid.layers.embedding(
        input=data, size=[input_dim, emb_dim], is_sparse=True)

    fc1 = fluid.layers.fc(input=emb, size=hid_dim)
    lstm1, cell1 = fluid.layers.dynamic_lstm(input=fc1, size=hid_dim)

    inputs = [fc1, lstm1]

    for i in range(2, stacked_num + 1):
        fc = fluid.layers.fc(input=inputs, size=hid_dim)
        lstm, cell = fluid.layers.dynamic_lstm(
            input=fc, size=hid_dim, is_reverse=(i % 2) == 0) #设置奇数层正向, 偶数层反向
        inputs = [fc, lstm]

    fc_last = fluid.layers.sequence_pool(input=inputs[0], pool_type='max')
    lstm_last = fluid.layers.sequence_pool(input=inputs[1], pool_type='max')

    prediction = fluid.layers.fc(
        input=[fc_last, lstm_last], size=class_dim, act='softmax')
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)
    return avg_cost, acc, prediction

# 训练函数
def train(train_reader, word_dict, network, use_cuda, save_dirname, lr=0.001, batch_size=16, pass_num=10):
    #输入层
    data = fluid.layers.data(name="words", shape=[1], dtype="int64", lod_level=1)
    #标签层
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")
    #网络结构
    cost, acc, prediction = network(data, label, len(word_dict))

    # 使用Adam优化器
    sgd_optimizer = fluid.optimizer.Adam(learning_rate=lr)
    sgd_optimizer.minimize(cost)
    # 设备、执行器、feeder 定义
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(feed_list=[data, label], place=place)
    # 模型参数初始化
    exe.run(fluid.default_startup_program())

    # 保存训练过程中的损失值、准确率、精确度、召回率、F1分数
    loss_values = []
    acc_values = []
    precision_values = []
    recall_values = []
    f1_values = []
    learning_rate_values = []
    y_true = []
    y_score = []

    for pass_id in range(pass_num):
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        all_labels = []
        all_preds = []

        for data in train_reader():
            avg_cost_np, avg_acc_np = exe.run(fluid.default_main_program(),
                                              feed=feeder.feed(data),
                                              fetch_list=[cost, acc])
            epoch_loss += np.mean(avg_cost_np)
            epoch_acc += np.mean(avg_acc_np)
            num_batches += 1

            pred = exe.run(fluid.default_main_program(),
                           feed=feeder.feed(data),
                           fetch_list=[prediction])[0]
            labels = [d[1] for d in data]
            all_labels.extend(labels)
            all_preds.extend(np.argmax(pred, axis=1))

            # 记录用于绘制ROC曲线的数据
            y_true.extend(labels)
            y_score.extend(pred)

        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        loss_values.append(avg_loss)
        acc_values.append(avg_acc)

        # 计算精确度、召回率和F1分数
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

        # 保存学习率（这里只是示例，实际上学习率可能是恒定的或者通过调度器变化）
        learning_rate_values.append(lr)

        if pass_id % 1 == 0:
            print("Pass {:d}, loss {:.6f}, accuracy {:.6f}".format(pass_id, avg_loss, avg_acc))

        # 保存模型
        epoch_model = save_dirname
        fluid.io.save_inference_model(epoch_model, ["words", "label"], acc, exe)

    # 绘制损失值、准确率、精确度、召回率、F1分数曲线
    plt.figure()
    #损失值
    plt.plot(range(pass_num), loss_values, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig("C:\\Users\\ljx20\\Documents\\code\\image\\loss_curve.png")
    plt.close()
    #准确度
    plt.plot(range(pass_num), acc_values, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.savefig("C:\\Users\\ljx20\\Documents\\code\\image\\accuracy_curve.png")
    plt.close()
    #精确度
    plt.plot(range(pass_num), precision_values, label='Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Precision Curve')
    plt.legend()
    plt.savefig("C:\\Users\\ljx20\\Documents\\code\\image\\precision_curve.png")
    plt.close()
    #召回率
    plt.plot(range(pass_num), recall_values, label='Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Recall Curve')
    plt.legend()
    plt.savefig("C:\\Users\\ljx20\\Documents\\code\\image\\recall_curve.png")
    plt.close()

    plt.plot(range(pass_num), f1_values, label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Curve')
    plt.legend()
    plt.savefig("C:\\Users\\ljx20\\Documents\\code\\image\\f1_score_curve.png")
    plt.close()

    # 绘制精确率-召回率曲线
    precision, recall, _ = precision_recall_curve(y_true, np.array(y_score)[:, 1])
    plt.figure(figsize=(10, 5))
    plt.plot(recall, precision, marker='o', linestyle='--', color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig("C:\\Users\\ljx20\\Documents\\code\\image\\precision_recall_curve.png")
    plt.close()

    # 绘制ROC曲线
    y_score = np.array(y_score)
    if y_score.shape[1] > 1:
        fpr, tpr, _ = roc_curve(y_true, y_score[:, 1], pos_label=1)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(10, 5))
        plt.plot(fpr, tpr, color='b', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig("C:\\Users\\ljx20\\Documents\\code\\image\\roc_curve.png")
        plt.close()
    else:
        print("Cannot compute ROC curve; insufficient score dimensions.")

    print('Training end')

    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig("C:\\Users\\ljx20\\Documents\\code\\image\\confusion_matrix.png")
    plt.close()

# 训练模型
train(
    train_reader,
    word_dict,
    lstm_net,
    use_cuda=False,
    save_dirname="lstm_model",
    lr=0.001,
    pass_num=10,
    batch_size=16)
#定义测试过程
def infer(test_reader, use_cuda, model_path=None):
    # 输入层
    data = fluid.layers.data(
        name="words", shape=[1], dtype="int64", lod_level=1)

    # 标签层
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")

    # 设置设备 和 执行器
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(feed_list=[data, label], place=place)

    # 创建并使用 scope
    inference_scope = fluid.core.Scope()

    with fluid.scope_guard(inference_scope):
        # 加载预测模型
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(model_path, exe)
        total_acc = 0.0
        total_count = 0
        for data in test_reader():
            # 预测
            acc = exe.run(inference_program,
                          feed=feeder.feed(data),
                          fetch_list=fetch_targets,
                          return_numpy=True)
            total_acc += acc[0] * len(data)
            total_count += len(data)

        avg_acc = total_acc / total_count
        print("model_path: %s, avg_acc: %f" % (model_path, avg_acc))
#实施预测
model_path = "lstm_model"
infer(test_reader, use_cuda=False, model_path=model_path)
