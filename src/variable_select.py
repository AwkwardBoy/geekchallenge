# -*- coding: utf-8 -*-
from utils import Columns
import pandas as pd


class VariableSelect:

    def __init__(self):
        self.train_pos = './game/clean_data/train_pos.csv'
        self.train_neg = './game/clean_data/train_neg.csv'
        self.eff_test = './game/clean_data/eff_test_v4.csv'

    # 合并loan_amnt, funded_amnt, funded_amnt_inv,
    @staticmethod
    def merge_load(filein):
        df = pd.read_csv(filein, low_memory=False)
        df['avg_loan'] = df[[Columns.loan_amnt, Columns.funded_amnt, Columns.funded_amnt_inv]].mean(axis=1)
        df.to_csv(filein, index=False)
    # 连续变量变化值

    def continuous_para(self, filein):
        df = pd.read_csv(filein, low_memory=False)
        df[Columns.collection_recovery_fee] = \
            pd.Series(map(self.collection_recovery_fee, df[Columns.collection_recovery_fee]))

        df[Columns.total_rec_late_fee] = \
            pd.Series(map(self.total_rec_late_fee, df[Columns.total_rec_late_fee]))

        df.to_csv(filein, index=False)

    #  连续变量离散 total_rec_late_fee, collection_recovery_fee
    @staticmethod
    def total_rec_late_fee(fee):
        if fee == 0:
            return 0

        if 0 < fee <= 10:
            return 1

        if 10 < fee <= 50:
            return 2

        if fee > 50:
            return 3

    @staticmethod
    def collection_recovery_fee(fee):
        if fee == 0:
            return 0

        if 0 < fee <= 10:
            return 1

        if 10 < fee <= 200:
            return 2

        if fee > 200:
            return 3


if __name__ == '__main__':
    vs = VariableSelect()

    vs.merge_load(vs.train_pos)
    vs.merge_load(vs.train_neg)
    vs.merge_load(vs.eff_test)

    vs.continuous_para(vs.train_pos)
    vs.continuous_para(vs.train_neg)
    vs.continuous_para(vs.eff_test)
