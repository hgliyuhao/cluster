
import os
# os.environ['RECOMPUTE']= "1"
import json
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm
import pylcs
from keras.layers import Dropout, Dense

# 基本信息
maxlen = 128
epochs = 13
batch_size = 32
learning_rate = 4e-5
crf_lr_multiplier = 100  # 必要时扩大CRF层的学习率


config_path = 'chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'chinese_L-12_H-768_A-12/vocab.txt'

# 读取schema
with open('event_schema/event_schema.json') as f:
    id2label, label2id, n = {}, {}, 0
    num_count = {}
    classify_id2label,classify_label2id, m = {}, {}, 0 
    for l in f:
        l = json.loads(l)
        for role in l['role_list']:
            key = (l['event_type'], role['role'])
            id2label[n] = key
            label2id[key] = n
            num_count[key] = 0
            n += 1
        for i in l:
            classify = l['event_type']
            classify = classify[:classify.find('-')]
            if classify not in classify_label2id:
                classify_id2label[m] = classify
                classify_label2id[classify] = m
                m += 1       
    num_labels = len(id2label) * 2 + 1
    classify_num_labels = len(classify_label2id)
def load_data(filename):
    D = []
    with open(filename) as f:
        for l in f:
            l = json.loads(l)
            arguments = {}
            for event in l['event_list']:
                classify = event['event_type']
                # '找到-前的部分'
                classify = classify[:classify.find('-')]
                num = classify_label2id[classify]
            D.append((l['text'], num))
    return D


vaild_data = load_data('train_data/train.json')
train_data = load_data('dev_data/dev.json')


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, num) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
            labels = [0] * len(token_ids)

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([num])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


bert = build_transformer_model(

    config_path=config_path,

    checkpoint_path=checkpoint_path,

    with_pool=True,

    return_keras_model=False,

)


classify_output = Dropout(rate=0.1)(bert.model.output)
classify_output = Dense(units=classify_num_labels,
                activation='softmax',
                name='classify_output',
                kernel_initializer=bert.initializer
                )(classify_output)

model = keras.models.Model(bert.model.input, classify_output)
model.summary()

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate),
    metrics=['accuracy'],

)

def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]

        for i in range(len(y_true)):
            if y_pred[i] == y_true[i]:
                right += 1
        total += len(y_true)
    return right / total


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(vaild_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model1.weights')
        print(val_acc)
        print(self.best_val_acc)

def adversarial_training(model, embedding_name, epsilon=1):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    # 封装为函数
    inputs = (model._feed_inputs +
              model._feed_targets +
              model._feed_sample_weights)  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):  # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs

    model.train_function = train_function  # 覆盖原训练函数



def search_layer(inputs, name, exclude=None):

    """根据inputs和name来搜索层
    说明：inputs为某个层或某个层的输出；name为目标层的名字。
    实现：根据inputs一直往上递归搜索，直到发现名字为name的层为止；
    如果找不到，那就返回None。

    """

    if exclude is None:
        exclude = set()

    if isinstance(inputs, keras.layers.Layer):
        layer = inputs
    else:
        layer = inputs._keras_history[0]

    if layer.name == name:
        return layer
    elif layer in exclude:
        return None
    else:
        exclude.add(layer)
        inbound_layers = layer._inbound_nodes[0].inbound_layers
        if not isinstance(inbound_layers, list):
            inbound_layers = [inbound_layers]
        if len(inbound_layers) > 0:
            for layer in inbound_layers:
                layer = search_layer(layer, name, exclude)
                if layer is not None:
                    return layer




if __name__ == '__main__':

    train_generator = data_generator(train_data, batch_size)
    vaild_generator = data_generator(vaild_data, batch_size)
    evaluator = Evaluator()

    # adversarial_training(model, 'Embedding-Token', 0.2)
    
    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        # class_weight = 'auto',
        callbacks=[evaluator]
    )


else:

    model.load_weights('best_model.weights')
    # predict_to_file('/root/baidu/datasets/ee/test1_data/test1.json', 'ee_pred.json')
