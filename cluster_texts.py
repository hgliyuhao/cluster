from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn.externals import joblib
import numpy as np
import random
import json
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn import metrics
import copy

def evaluate(predict):

    # 评估准去率的函数
    f1 = 'D:/cluster/data/train.json'
    classify  = load_data(f1)

    classify2id = {}
    for i in range(9):
        classify2id[i] = []

    for num,c in enumerate(classify):
        classify2id[c].append(num)

    final_res = {}
    for i in classify2id:
        res = {}
        for ids in classify2id[i]:
            if predict[ids] not in res:
                res[predict[ids]]  = 1
            else:
                res[predict[ids]] += 1   
        res = sorted(res.items(),key=lambda x:x[1],reverse=True)
        final_res[i] = res
        print(res)
    res2id = {}
    for i in range(9):
        res2id[i] = 0
    for i in final_res:
        classify_num = final_res[i][0][0]
        num = final_res[i][0][1]
        if classify_num not in res2id:
            res2id[classify_num] = num
        else:
            if res2id[classify_num] < num:
                res2id[classify_num] = num
    right = 0
    for i in res2id:
        # print(res2id[classify_num])
        right += res2id[i]
    print(res2id)
    print(len(classify))
    acc = right / len(classify)    
    print(acc)
    return acc

def load_data(filename):

    classify = ['财经/交易','产品行为','交往','竞赛行为','人生','司法行为','灾害/意外','组织行为','组织关系']
    D = []
    num = 0
    with open(filename,encoding='utf-8') as f:
        for l in f:
            num += 1
            l = json.loads(l)

            classify_name = l['event_list'][0]['class']

            for i ,c in enumerate(classify):
                if c in classify_name:
                    D.append(i)
                    break          
    return D

if __name__ == "__main__":
    
    # 读取提取的特征
    feature = np.loadtxt("data/funtune-cls-val.txt")

    # k-means 聚类

    for i in range(6):
        clf = KMeans(n_clusters=9)
        s = clf.fit(feature)
        pre = clf.predict(feature)
        print(evaluate(pre))

    # birth聚类
   
    lists = [10,25]
    best_score = 0
    best_i = -1
    for i in lists:
        print(i)
        y_pred = Birch(branching_factor=i, n_clusters = 9, threshold=0.5,compute_labels=True).fit_predict(feature)
        score = evaluate(y_pred)
        if score > best_score:
            best_score = score
            best_i = i
        print(metrics.calinski_harabaz_score(feature, y_pred)) 
        print(best_score)
        print(best_i)
