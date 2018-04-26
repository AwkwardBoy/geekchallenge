# -*- coding: utf-8 -*-
import xgboost as xgb
import numpy as np
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


class Merge():
    def __init__(self):

        data_train = np.load('../game/clean_data/prediction_probs.npy')
        self.data_train = np.transpose(data_train)
        self.train_labels = np.hstack((np.ones(2500), np.zeros(550000)))
        data_test = np.load('../game/clean_data/test_prediction_probs.npy')
        self.data_test = np.transpose(data_test)
        self.test_labels = np.hstack((np.ones(793), np.zeros(self.data_test.shape[0] - 793)))

    def xgb_merge(self):

        data_train = xgb.DMatrix(data=self.data_train, label=self.train_labels)
        data_test = xgb.DMatrix(data=self.data_test)
        para = {
            'max_depth': 5, 'eta': 0.01, 'silent': 0, 'objective': 'binary:logistic',
            'subsample': 0.7, 'scale_pos_weight': 200
        }

        boost = xgb.train(params=para, dtrain=data_train, num_boost_round=20)
        xgb.plot_importance(boost, max_num_features=30)
        plt.show()
        train_preds = boost.predict(xgb.DMatrix(data=self.data_train))

        test_preds = boost.predict(data_test)
        train_preds = np.array(list(map(lambda x: int(x > 0.5), train_preds)))
        test_preds = np.array(list(map(lambda x: int(x > 0.5), test_preds)))

        print(classification_report(y_true=self.train_labels, y_pred=train_preds))
        print(classification_report(y_true=self.test_labels, y_pred=test_preds))

    def lr_merge(self):

        lr = LogisticRegression(penalty='l1', C=1, max_iter=10)
        lr.fit(self.data_train, self.train_labels)
        train_preds = lr.predict(self.data_train)
        test_preds = lr.predict(self.data_test)
        print(lr.coef_)

        print(classification_report(y_true=self.train_labels, y_pred=train_preds))
        print(classification_report(y_true=self.test_labels, y_pred=test_preds))


if __name__ == '__main__':
    m = Merge()
    m.xgb_merge()
    # m.lr_merge()
