# -*- coding: utf-8 -*-
import numpy as np
from src.sampling import Sampling
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import fbeta_score


class ModelSelect:

    def __init__(self):
        self.s = Sampling()
        train_pos = self.s.data_load('../game/clean_data/train_pos.csv')
        train_neg = self.s.data_load('../game/clean_data/train_neg.csv')
        self.train_pos = np.array(train_pos)
        self.train_neg = np.array(train_neg)

    @staticmethod
    def data_stack(t_pos, t_neg):
        data = np.vstack((t_pos, t_neg))
        labels = np.hstack((np.ones(t_pos.shape[0]), np.zeros(t_neg.shape[0])))
        return data, labels

    def train_sampling(self):
        t_pos = self.train_pos
        t_neg = self.s.random_sample(self.train_neg, 5000, False)
        print(t_pos.shape)
        print(t_neg.shape)

        t_train, t_labels = self.data_stack(t_pos, t_neg)
        t_train_sample, t_label_sample = self.s.smote_sampling(t_train, t_labels)
        print(t_train_sample.shape, t_label_sample.shape)

        return t_train_sample, t_label_sample

    def tune_lgb(self):
        para = {
            'num_leaves': 50,
            'max_bin': 300,
            'learning_rate': 0.1,
            'max_depth': 8,
            'objective': 'binary',
            'min_data_in_leaf': 100,
            'subsample': 1,
            'is_unbalance': True,
            'feature_fraction': 0.8,
            'min_split_gain': 0.2,
            'lambda_l1': 0.,
            'lambda_l2': 0.,

        }
        train_set, test_set = self.data_stack(self.train_pos, self.train_neg)
        # t_train_sample, t_label_sample = self.train_sampling()

        kf = KFold(n_splits=5, shuffle=True)

        train_set, test_set, train_labels, test_labels = \
            train_test_split(train_set, test_set, test_size=0.2, random_state=1)

        f2_score = 0
        for train, test in kf.split(train_set):

            train_train_set = lgb.Dataset(data=train_set[train], label=train_labels[train])
            lgb_classifier = lgb.train(para, train_train_set, 500)
            preds = lgb_classifier.predict(train_set[test])
            preds = list(map(lambda a: float(a > 0.5), preds))
            trues = train_labels[test]
            f2_score += fbeta_score(y_true=trues, y_pred=preds, beta=2)

        print(f2_score / 5)

        # lgb_classifier.save_model('../model/lgb_model/model_' + str(k) + '.lgb')
        # predict_probs = lgb_classifier.predict(test_set)
        # predictions = list(map(lambda a: float(a > 0.5), predict_probs))
        # print(fbeta_score(y_pred=predictions, y_true=test_labels, beta=2))

    def tune_xgb(self, k):
        para = {
            'max_depth': 8,
            'eta': 0.01,
            'silent': 0,
            'objective': 'binary:logistic',
            'subsample': 0.75,
            'scale_pos_weight': 1
        }
        t_train_sample, t_label_sample = self.train_sampling()

        # train_set, test_set, train_labels, test_labels = \
        #     train_test_split(t_train_sample, t_label_sample, test_size=0.2, random_state=1)

        train_set = xgb.DMatrix(data=t_train_sample, label=t_label_sample)
        # test_set = xgb.DMatrix(data=test_set)
        xgb_classifier = xgb.train(para, train_set, 500)
        xgb_classifier.save_model('../model/xgb_model/model_' + str(k) + '.xgb')

        # predict_probs = xgb_classifier.predict(test_set)

        # predictions = list(map(lambda a: float(a > 0.5), predict_probs))
        # print(classification_report(y_pred=predictions, y_true=test_labels))

    def tune_svm(self):

        svm_classifier = LinearSVC(C=0.01, penalty='l2', class_weight='balanced', tol=1e-4)
        t_train_sample, t_label_sample = self.train_sampling()

        train_set, test_set, train_labels, test_labels = \
            train_test_split(t_train_sample, t_label_sample, test_size=0.2, random_state=1)

        svm_classifier.fit(train_set, train_labels)
        predictions = svm_classifier.predict(test_set)

        print(classification_report(y_pred=predictions, y_true=test_labels))


if __name__ == '__main__':

    ms = ModelSelect()
    ms.tune_lgb()
    # for i in range(30):
    #     ms.tune_lgb(i)

    # for j in range(30):
    #     ms.tune_xgb(j)

    # ms.tune_svm()
