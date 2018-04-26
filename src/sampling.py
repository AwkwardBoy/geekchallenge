# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from src.utils import Columns
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from scipy import stats
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

FEATURES = [
    Columns.avg_loan, Columns.term, Columns.int_rate, Columns.installment, Columns.grade,
    Columns.sub_grade, Columns.emp_length, Columns.home_ownership, Columns.annual_inc, Columns.verification_status,
    Columns.issue_d, Columns.loan_status, Columns.dti, Columns.earliest_cr_line, Columns.pub_rec,
    Columns.revol_bal, Columns.revol_util, Columns.total_acc, Columns.initial_list_status, Columns.out_prncp,
    Columns.out_prncp_inv, Columns.total_pymnt, Columns.total_pymnt_inv, Columns.total_rec_prncp, Columns.total_rec_int,
    Columns.recoveries, Columns.purpose, Columns.collections_12_mths_ex_med, Columns.lat, Columns.lng,
    Columns.tot_coll_amt, Columns.tot_cur_bal, Columns.total_rev_hi_lim, Columns.rev_rel, Columns.dti_joint,
    Columns.annual_inc_joint, Columns.application_type, Columns.policy_code, Columns.pymnt_plan,
    Columns.verification_status_joint, Columns.collection_recovery_fee
]

CONTINUOUS = [
    Columns.avg_loan, Columns.installment, Columns.annual_inc, Columns.issue_d,
    Columns.earliest_cr_line, Columns.revol_bal, Columns.revol_util, Columns.out_prncp, Columns.total_rec_prncp,
    Columns.total_pymnt, Columns.tot_coll_amt, Columns.recoveries, Columns.tot_cur_bal
]

IMPORTANT_FEATURES = [
    Columns.avg_loan, Columns.term, Columns.int_rate, Columns.installment, Columns.grade,
    Columns.tot_cur_bal, Columns.total_rev_hi_lim, Columns.annual_inc, Columns.issue_d, Columns.recoveries,
    Columns.purpose, Columns.total_rec_late_fee, Columns.verification_status, Columns.total_rec_prncp, Columns.dti,
    Columns.collections_12_mths_ex_med, Columns.lat, Columns.lng, Columns.tot_coll_amt, Columns.pub_rec
]

# xgb 挑选变量
OUT = [
    Columns.rev_rel, Columns.dti_joint, Columns.annual_inc_joint, Columns.application_type, Columns.policy_code,
    Columns.pymnt_plan, Columns.verification_status_joint, Columns.collection_recovery_fee
]


class Sampling:

    def __init__(self):
        self.clean_train = '../game/clean_data/eff_train_v4.csv'
        self.train_pos = '../game/clean_data/train_pos.csv'
        self.train_neg = '../game/clean_data/train_neg.csv'

    def xgb_sampling(self, neg_filein, max_dep, no):
        subsample = np.random.random() / 10 + 0.7
        t_pos = self.data_load(self.train_pos)
        t_neg = self.data_load(neg_filein)
        t_pos = np.array(t_pos)[0:2500]
        t_neg = np.array(t_neg)

        if t_neg.shape[0] >= train_pos.shape[0]:
            t_neg = s.random_sample(t_neg, t_pos.shape[0], replace=False)

        para = {
            'max_depth': max_dep, 'eta': 0.1, 'silent': 0, 'objective': 'binary:logistic',
            'subsample': subsample, 'scale_pos_weight': t_neg.shape[0] / t_pos.shape[0]
        }

        labels = np.hstack((np.ones(t_pos.shape[0]), np.zeros(t_neg.shape[0])))
        data = np.vstack((t_pos, t_neg))

        # t_train, t_test, t_train_labels, t_test_labels = train_test_split(data, labels, test_size=0.1, shuffle=True)

        # self.xgboost_classifier(t_train, t_test, t_train_labels, t_test_labels, 15, para)
        #
        dtrain = xgb.DMatrix(data=data, label=labels)
        boost = xgb.train(params=para, dtrain=dtrain, num_boost_round=10)
        y_pred = boost.predict(xgb.DMatrix(data=data))

        y_pred = np.array(list(map(lambda x: int(x > 0.5), y_pred)))
        print(classification_report(y_true=labels, y_pred=y_pred))
        boost.save_model('../model/xgb_model/sampling_' + str(no) + '.xgb')

    def step_wise_sampling(self):
        pass

    @staticmethod
    def data_load(file_in):
        dat = pd.read_csv(file_in, low_memory=False)
        dat = dat[FEATURES]
        return dat

    # 随机采样
    @staticmethod
    def random_sample(arr, num, replace):
        n, p = arr.shape
        ind = np.random.choice(range(n), num, replace=replace)
        return arr[ind]

    @staticmethod
    def xgboost_classifier(train, valid, train_labels, valid_labels, num_round, param):

        dtrain = xgb.DMatrix(data=train, label=train_labels)
        dvalid = xgb.DMatrix(data=valid)
        bst = xgb.train(params=param, dtrain=dtrain, num_boost_round=num_round)
        y_pred = bst.predict(dvalid)

        print(roc_auc_score(y_true=valid_labels, y_score=y_pred))
        y_pred = np.array(list(map(lambda x: int(x > 0.5), y_pred)))

        print(classification_report(y_true=valid_labels, y_pred=y_pred))

        return bst

    def remains(self, xgb_model, pre_remains, no):
        xgbst = xgb.Booster()
        xgbst.load_model(xgb_model)
        remains = self.data_load(pre_remains)
        features = np.array(remains)
        features = xgb.DMatrix(data=features)

        preds = xgbst.predict(features)
        preds = np.array(list(map(lambda x: int(x > 0.5), preds)))
        remains['pred'] = pd.Series(preds)

        next_remains = remains[remains['pred'] != 0][FEATURES]
        print(next_remains.describe())
        next_remains.to_csv('./game/remains/remains_' + str(no) + '.csv', index=False)

    @staticmethod
    def smote_sampling(x, y):
        sm = SMOTE(k_neighbors=3, m_neighbors=10, kind='borderline1')
        x_resampled, y_resampled = sm.fit_sample(x, y)

        return x_resampled, y_resampled


if __name__ == '__main__':

    s = Sampling()

    # data_load
    train_pos = s.data_load(s.train_pos)
    train_neg = s.data_load(s.train_neg)
    train_pos = np.array(train_pos)
    train_neg = np.array(train_neg)

    # stack data and make labels
    data = np.vstack((train_pos, train_neg))
    true_labels = np.hstack((np.ones(train_pos.shape[0]), np.zeros(train_neg.shape[0])))

    # SMOTE TEST

    # s.smote_sampling(data, true_labels)

    # 逐步欠采样

    # para = {
    #     'max_depth': 5, 'eta': 0.1, 'silent': 0, 'objective': 'binary:logistic',
    #     'subsample': 0.7, 'scale_pos_weight': 250
    # }

    # labels = np.hstack((np.ones(train_pos.shape[0]), np.zeros(train_neg.shape[0])))
    # data = np.vstack((train_pos, train_neg))
    # train_set, valid_set, t_labels, v_labels = \
    #     train_test_split(data, labels, test_size=0.2, shuffle=True)

    # sample
    # para = {
    #     'max_depth': 10, 'eta': 0.1, 'silent': 0, 'objective': 'binary:logistic',
    #     'subsample': 0.7, 'scale_pos_weight': 3
    # }

    # train_neg_sample = s.random_sample(train_neg, 10000, replace=False)
    # labels = np.hstack((np.ones(train_pos.shape[0]), np.zeros(train_neg_sample.shape[0])))
    # data = np.vstack((train_pos, train_neg_sample))
    # train_samples, valid_samples, t_labels, v_labels = \
    #     train_test_split(data, labels, test_size=0.2, random_state=0)

    # para = {
    #     'max_depth': 8, 'eta': 0.1, 'silent': 0, 'objective': 'binary:logistic',
    #     'subsample': 0.7, 'scale_pos_weight': 3
    # }

    # para = {
    #     'max_depth': 5, 'eta': 0.1, 'silent': 0, 'objective': 'binary:logistic',
    #     'subsample': 0.7, 'scale_pos_weight': 3
    # }
    # bst = s.xgboost_classifier(train_samples, valid_samples, t_labels, v_labels, 15, param=para)
    #
    # bst.save_model('../model/xgb_model/sampling.xgb')

    # 逐步挑选变量
    # for i in range(200):
    #     s.xgb_sampling('../game/remains/remains.csv', 5, i)

    # data_test = np.vstack((train_pos, train_neg))
    # y_preds = np.zeros(len(trues))
    # data_test = xgb.DMatrix(data_test)
    #
    # predictions = []
    # prediction_probs = []
    #
    # for i in range(30):
    #     bst = xgb.Booster()
    #     bst.load_model('./model/xgb_model/sampling_' + str(i) + '.xgb')
    #
    #     prob = bst.predict(data_test)
    #
    #     y_preds += prob
    #
    # y_preds = y_preds / 30
    # y_preds = np.array(list(map(lambda x: float(x >= 0.5), y_preds)))
    # print(classification_report(y_true=trues, y_pred=y_preds))

    # predictions_probs = np.array(prediction_probs)
    # predictions = np.array(predictions)
    #
    # np.save('./game/clean_data/test_prediction_probs.npy', prediction_probs)
    # np.save('./game/clean_data/test_predictions.npy', predictions)


