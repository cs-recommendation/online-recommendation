# coding:utf-8
# author: lu yf
# create date: 2018/6/25

from __future__ import division
import os
import random
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score, f1_score, roc_auc_score, accuracy_score
from sklearn.cluster import KMeans
import warnings
import matplotlib.pyplot as plt
import scipy.io
from tqdm import tqdm
import numpy as np
from six.moves import xrange

warnings.filterwarnings('ignore')
random.seed(1)


class Evaluation:

    def __init__(self, embeddings_data):
        self.embeddings_data = embeddings_data
        self.name_emb_dict = {}

    def load_embeddings(self):
        embeddings_mat = scipy.io.loadmat(self.embeddings_data)
        key = filter(lambda k:k.startswith('_') is False,embeddings_mat.keys())[0]
        embeddings = embeddings_mat[key]
        for i in range(len(embeddings)):
            self.name_emb_dict[i] = embeddings[i]

    def kmean(self,cluster_k):
        x = []
        y = []
        with open('../data/dblp/author_label.txt', 'r') as author_name_label_file:
            author_name_label_lines = author_name_label_file.readlines()
        for line in author_name_label_lines:
            tokens = line.strip().split('\t')
            if self.name_emb_dict.has_key(int(tokens[0])):
                x.append(list(self.name_emb_dict[int(tokens[0])]))
                y.append(int(tokens[1]))
        km = KMeans(n_clusters=cluster_k)
        km.fit(x, y)
        y_pre = km.predict(x)
        nmi = normalized_mutual_info_score(y, y_pre)
        return nmi

    def classification(self,train_size):
        x = []
        y = []
        with open('../data/dblp/author_label.txt', 'r') as author_name_label_file:
            author_name_label_lines = author_name_label_file.readlines()
        for line in author_name_label_lines:
            tokens = line.strip().split('\t')
            if self.name_emb_dict.has_key(int(tokens[0])):
                x.append(list(self.name_emb_dict[int(tokens[0])]))
                y.append(int(tokens[1]))
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=1-train_size,random_state=9)
        lr = LogisticRegression()
        lr.fit(x_train, y_train)
        y_valid_pred = lr.predict(x_valid)
        micro_f1 = f1_score(y_valid, y_valid_pred,average='micro')
        macro_f1 = f1_score(y_valid, y_valid_pred,average='macro')
        return macro_f1,micro_f1

    def calculate_sim(self,u,v,sum_flag):
        if sum_flag:
            return sum(np.abs(np.array(u)-np.array(v)))
        else:
            return np.abs(np.array(u)-np.array(v))

    def binary_classification(self, x_train, y_train, x_test, y_test):
        classifier = LogisticRegression()
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict_proba(x_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred)
        f1 = f1_score(y_test, classifier.predict(x_test))
        acc = accuracy_score(y_test,classifier.predict(x_test))
        print('auc: {}'.format(auc_score))
        print('f1: {}'.format(f1))
        print('acc: {}'.format(acc))

    def link_prediction(self,data_type):
        x = []
        y = []
        with open('../data/dblp_temp/coauthor'+data_type, 'r') as p_co_file:
            for line in p_co_file:
                tokens = line.strip().split('\t')
                a1_name = int(tokens[0])
                a2_name = int(tokens[1])
                if self.name_emb_dict.has_key(a1_name) and self.name_emb_dict.has_key(a2_name):
                    if tokens[0] in self.good_author and tokens[1] in self.good_author:
                        a1_emb = self.name_emb_dict[a1_name]
                        a2_emb = self.name_emb_dict[a2_name]
                        sim_a1_a2 = self.calculate_sim(a1_emb, a2_emb, sum_flag=False)
                        x.append(sim_a1_a2)
                        y.append(1)
        with open('../data/dblp_temp/neg_coauthor_' + data_type, 'r') as p_co_file:
            for line in p_co_file:
                tokens = line.strip().split('\t')
                a1_name = int(tokens[0])
                a2_name = int(tokens[1])
                if self.name_emb_dict.has_key(a1_name) and self.name_emb_dict.has_key(a2_name):
                    if tokens[0] in self.good_author and tokens[1] in self.good_author:
                        a1_emb = self.name_emb_dict[a1_name]
                        a2_emb = self.name_emb_dict[a2_name]
                        sim_a1_a2 = self.calculate_sim(a1_emb, a2_emb, sum_flag=False)
                        x.append(sim_a1_a2)
                        y.append(0)
        return x,y

    def link_prediction_with_auc(self):
        train_x, train_y = self.link_prediction('train')
        test_x, test_y = self.link_prediction('test')
        self.binary_classification(train_x, train_y, test_x, test_y)
        x_train, x_valid, y_train, y_valid = train_test_split(test_x, test_y, test_size=1 - 0.8, random_state=9)
        self.binary_classification(x_train, y_train, x_valid, y_valid)

    def node_recommendation(self,hit_k):
        with open('../data/dblp_temp1/test.txt', 'r') as test_ac_p_n_f:
            ac_lines = test_ac_p_n_f.readlines()
        hit_or_not = []
        for aa in list(set(ac_lines)):
            ac_list = aa.strip().split('\t')
            pos_a = ac_list[0]
            pos_c = ac_list[1]
            if self.name_emb_dict.has_key('a'+pos_a) and self.name_emb_dict.has_key('c'+pos_c):
                neg_c_list = ac_list[2].split(' ')
                pos_a_embs = self.name_emb_dict['a' + pos_a]
                pos_c_embs = self.name_emb_dict['c' + pos_c]
                pos_sim = self.calculate_sim(pos_a_embs, pos_c_embs, sum_flag=True)
                neg_sim_list = map(lambda x:
                                   self.calculate_sim(pos_a_embs, self.name_emb_dict['c' + x], sum_flag=True),
                                   neg_c_list)
                neg_c_list.append(pos_c)
                neg_sim_list.append(pos_sim)
                sim_dict = dict(zip(neg_c_list, neg_sim_list))
                sorted_sim_triple = sorted(sim_dict.iteritems(), key=lambda d: d[1])
                if (pos_c, pos_sim) in sorted_sim_triple[:hit_k]:
                    hit_or_not.append(1)
                else:
                    hit_or_not.append(0)
        print('#test: {}'.format(len(hit_or_not)))
        print('hit@{}: {}'.format(hit_k, sum(hit_or_not) / len(hit_or_not)))

    def get_good_author(self):
        id2node = {}
        self.good_author = []
        with open('../baseline/dblp_temp3/node','r') as d_f:
            for line in d_f:
                tokens = line.strip().split('\t')
                id2node[tokens[1]] = tokens[0]
        with open('../baseline/dblp_temp3/edgelist','r') as f:
            for line in f:
                tokens = line.strip().split(' ')
                if id2node[tokens[0]].startswith('a'):
                    self.good_author.append(id2node[tokens[0]][1:])
                if id2node[tokens[1]].startswith('a'):
                    self.good_author.append(id2node[tokens[1]][1:])
        self.good_author = list(set(self.good_author))
        print(len(self.good_author))

if __name__ == '__main__':

    train_ratio = [0.2,0.4,0.6,0.8]
    embeddings_data = '../data/dblp/embedding.mat'
    print(embeddings_data)
    exp = Evaluation(embeddings_data)
    exp.load_embeddings()
    exp.kmean(cluster_k=4)
    for t_r in train_ratio:
        print(t_r)
        exp.classification(train_size=t_r)
    embeddings_data = '../data/dblp_temp/embedding_dblp.mat'
    print(embeddings_data)
    exp = Evaluation(embeddings_data)
    exp.get_good_author()
    exp.load_embeddings()
    exp.link_prediction_with_auc()
    for t_r in train_ratio:
        print(t_r)
        ma_f1 = []
        mi_f1 = []
        nmi = []
        for i in xrange(10):
            embeddings_data = '../data/dblp_temp4/'+str(i)+'_embedding.mat'
            print(embeddings_data)
            exp = Evaluation(embeddings_data)
            exp.load_embeddings()
            nmi_tmp = exp.kmean(cluster_k=4)
            ma_f1_tmp, mi_f1_tmp = exp.classification(train_size=t_r)
            nmi.append(nmi_tmp)
            ma_f1.append(ma_f1_tmp)
            mi_f1.append(mi_f1_tmp)
        print('ave. nim: {}'.format(sum(nmi) / 10))
        print('ave. ma_f1: {}'.format(sum(ma_f1) / 10))
        print('ave. mi_f1: {}'.format(sum(mi_f1) / 10))
    embeddings_data = '../data/dblp_temp3/embedding.mat'
    print(embeddings_data)
    exp = Evaluation(embeddings_data)
    exp.get_good_author()
    exp.load_embeddings()
    exp.link_prediction_with_auc()
    embeddings_data = '../data/dblp/embedding1.mat'
    exp = Evaluation(embeddings_data)
    exp.load_embeddings()