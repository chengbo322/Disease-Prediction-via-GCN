import enum
from sklearn import svm
import sklearn
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import heapq
from sklearn.tree import DecisionTreeClassifier
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve, auc
import torch
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier 

file_path = "./data/sample_data/sample_garph"
node_list = pickle.load(open(file_path + ".nodes.pkl", "rb"))
adj_lists = pickle.load(open(file_path + ".adj.pkl", "rb"))
rare_patient = pickle.load(open(file_path + ".rare.label.pkl", "rb"))
labels = pickle.load(open(file_path + ".label.pkl", "rb"))
node_map = pickle.load(open(file_path + ".map.pkl", "rb"))
train = pickle.load(open(file_path + ".train.pkl", "rb"))
test = pickle.load(open(file_path + ".test.pkl", "rb"))

symptom_num = 131
symptom_start_num = 877
patient_end_num = 805
cutoff = 5

train_set = []
train_labels = []
test_set = []
test_labels = []

def build_set(Set):
    X = []
    Y = []
    for id in Set:
        if id > patient_end_num and id < symptom_start_num:
            # print(id)
            continue
        neighbor = adj_lists[id]
        feature = [0]*symptom_num
        for nei in neighbor:
            feature[int(nei)-symptom_start_num] += 1
        id_labels = labels[id]
        X.append(feature)
        Y.append(id_labels)
    return np.array(X), np.array(Y)

def set_model(name):
    if name == "svm":
        model = OneVsRestClassifier(svm.SVC(C=0.1,kernel='linear',probability=True),n_jobs=-1)
    elif name == "dt":
        model = OneVsRestClassifier(DecisionTreeClassifier(),n_jobs=-1)
    elif name == "rf":
        model = OneVsRestClassifier(RandomForestClassifier(n_estimators = 15, criterion = 'entropy', random_state = 0),n_jobs=-1)
    elif name == "sgd":
        model = OneVsRestClassifier(make_pipeline(StandardScaler(),SGDClassifier(max_iter=1000, tol=1e-3,loss='log')),n_jobs=-1)
    elif name == "knn":
        model = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=10),n_jobs=-1)
    return model

def evaluate(data_name, val_output, test_labels, topk=(1, 2, 3, 4, 5,)):
    print("----" * 25)
    print()
    print("%s: " % data_name)

    # shape: batchnum * classnum
    target = torch.LongTensor(test_labels)
    output = val_output  # shape: batchnum * classnum

    print(target.shape, output.shape)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)

    # print(pred)
    correct = torch.zeros_like(pred)
    for i in range(batch_size):
        for k in range(maxk):
            correct[i, k] = 1 if target[i][pred[i, k]] == 1 else 0
    correct = correct.t()

    correct_target = target.sum(1, keepdim=True).squeeze().float()
    # print(correct_target)

    for k in topk:
        correct_k = correct[:k].sum(0, keepdim=True).squeeze().float()
        # print(correct_k)

        precision_k = 0.0
        # recall_k = 0.0
        for i in range(0, batch_size):
            # _k = k if k < correct_target[i].data else correct_target[i]
            _k = k
            precision_k += correct_k[i] / _k
            # recall_k += correct_k[i] / correct_target[i]
        precision_k = precision_k / batch_size
        # recall_k = recall_k / batch_size

        # print("precision @", k, precision_k.data)
        # print("recall @", k, recall_k.data)

        # precision_k = correct_k / k
        # precision_k = precision_k.sum() / batch_size

        recall_k = correct_k / correct_target
        recall_k = recall_k.sum() / batch_size

        f1_k = 2 * precision_k * recall_k / (precision_k + recall_k)

        print("precision @ %d : %.5f, recall @ %d : %.5f, f1 @ %d : %.5f" % (
            k, precision_k.data, k, recall_k.data, k, f1_k.data))
        # print("precision @", k, precision_k.data)
        # print("recall @", k, recall_k.data)
        # print("f1 @", k, f1_k.data)

        # print("precision@%d: %f" & (k, precision_k))
        # print("recall@%d: %f" & (k, recall_k))
    print()


train_set, train_labels = build_set(train)
test_set, test_labels = build_set(test)

model_name = "svm"
model = set_model(model_name)
model.fit(train_set, train_labels)
pre = (model.predict(test_set))
prob = model.predict_proba(test_set)

evaluate(model_name, torch.FloatTensor(prob), test_labels) 

model_name = "dt"
model = set_model(model_name)
model.fit(train_set, train_labels)
pre = (model.predict(test_set))
prob = model.predict_proba(test_set)

evaluate(model_name, torch.FloatTensor(prob), test_labels) 

model_name = "rf"
model = set_model(model_name)
model.fit(train_set, train_labels)
pre = (model.predict(test_set))
prob = model.predict_proba(test_set)

evaluate(model_name, torch.FloatTensor(prob), test_labels) 

model_name = "sgd"
model = set_model(model_name)
model.fit(train_set, train_labels)
pre = (model.predict(test_set))
prob = model.predict_proba(test_set)

evaluate(model_name, torch.FloatTensor(prob), test_labels) 

model_name = "knn"
model = set_model(model_name)
model.fit(train_set, train_labels)
pre = (model.predict(test_set))
prob = model.predict_proba(test_set)

evaluate(model_name, torch.FloatTensor(prob), test_labels)  