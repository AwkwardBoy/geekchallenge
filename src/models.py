# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from src.sampling import Sampling
from sklearn.metrics import classification_report
import xgboost as xgb
import lightgbm as lgb


class Models:

    def __init__(self):
        self.s = Sampling()
        x_pos = self.s.data_load('../game/clean_data/train_pos.csv')
        x_neg = self.s.data_load('../game/clean_data/train_neg.csv')
        x_pos = np.array(x_pos)
        x_neg = np.array(x_neg)
        # bootstrap_x_pos = x_pos[np.random.choice(range(2500), 500000, replace=True)]
        # bootstrap_x_neg = x_neg[np.random.choice(range(550000), 500000, replace=True)]
        # all_data = np.vstack((x_pos, x_neg))
        # all_labels = np.hstack((np.ones(x_pos.shape[0]), np.zeros(x_neg.shape[0])))

        self.train_set = np.vstack((x_pos, x_neg))
        self.train_labels = np.hstack((np.ones(x_pos.shape[0]), np.zeros(x_neg.shape[0])))
        # self.test_set = np.vstack((x_pos[2500:], x_neg[550000:]))
        # self.test_labels = np.hstack((np.ones(x_pos[2500:].shape[0]), np.zeros(x_neg[550000:].shape[0])))
        # self.train_set, self.test_set, self.train_labels, self.test_labels = \
        #     train_test_split(all_data, all_labels, test_size=0.2)

    def random_forests(self, n_estimators, max_depth, class_weight):
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight=class_weight,
                                    bootstrap=True)

        rf.fit(self.train_set, self.train_labels)

        print(rf.feature_importances_)
        y_preds = rf.predict(self.test_set)
        print(classification_report(y_pred=y_preds, y_true=self.test_labels))

    def lgb_classifier(self, max_depth, num_leaves, learning_rate, num_round, k):
        train_data = lgb.Dataset(self.train_set, label=self.train_labels)
        param = {'num_leaves': num_leaves, 'learning_rate': learning_rate, 'max_depth': max_depth,
                 'objective': 'binary', 'scale_pos_weight': 250, 'subsample': 0.75}

        lgb_classifier = lgb.train(param, train_data, num_boost_round=num_round)

        lgb_classifier.save_model('../model/lgb_model/big_model_' + str(k) + '.lgb')

        # y_pred = lgb_classifier.predict(self.test_set)
        # y_pred = np.array(list(map(lambda x: int(x > 0.5), y_pred)))
        # print(classification_report(y_true=self.test_labels, y_pred=y_pred))

    def xgb_classifier(self, max_dep, eta, num_round, k):
        dtrain = xgb.DMatrix(data=self.train_set, label=self.train_labels)
        # dvalid = xgb.DMatrix(data=self.test_set)
        para = {
            'max_depth': max_dep, 'eta': eta, 'silent': 0, 'objective': 'binary:logistic',
            'subsample': 0.75, 'scale_pos_weight': 1
        }
        bst = xgb.train(params=para, dtrain=dtrain, num_boost_round=num_round)
        bst.save_model('../model/lgb_model/big_model_' + str(k) + '.xgb')
        # y_pred = bst.predict(dvalid)
        # y_pred = np.array(list(map(lambda x: int(x > 0.5), y_pred)))
        #
        # print(classification_report(y_true=self.test_labels, y_pred=y_pred))


if __name__ == '__main__':
    m = Models()
    # m.random_forests(100, 3, 'balanced')
    for i in range(30):
        # m.xgb_classifier(5, 0.01, 500, i)
        m.lgb_classifier(8, 200, 0.1, 500, i)
