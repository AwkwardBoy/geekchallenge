# -*- coding: utf-8 -*-
from .utils import Columns
from .utils import Utils
import pandas as pd
from .sampling import IMPORTANT_FEATURES
from .sampling import CONTINUOUS
from collections import Counter
import numpy as np

u = Utils()

# 删除缺失变量过多的字段
def remove(filein, fileout, file_type):
    dat = pd.read_csv(filein, low_memory=False)
    member_id = dat[Columns.member_id]
    features = dat[Columns.features]
    features[Columns.member_id] = pd.Series(member_id)
    if file_type == 'train':
        labels = dat[Columns.label] = dat[Columns.label]
        features[Columns.label] = pd.Series(labels)
    features.to_csv(fileout, index=False)


# 填充变量处理变量
def process(filein, fileout):

    dat = pd.read_csv(filein, low_memory=False)

    dat = u.trans_time(dat)

    dat = u.handle_missing(dat)
    dat = u.discrete_encode(dat)
    dat.to_csv(fileout, index=False)


def scaled(filein, file_out):
    dat = pd.read_csv(file_in)
    for variable in CONTINUOUS:



if __name__ == '__main__':
    pass
    # dat = pd.read_csv('./game/train.csv')

    # 去除缺失变量
    # remove('../game/train.csv', '../game/clean_data/eff_train.csv', 'train')
    # remove('../game/test.csv', '../game/clean_data/eff_test.csv', 'test')

    # 时间戳转换
    # eff_train = pd.read_csv('../game/clean_data/eff_train.csv')
    # eff_train = u.trans_time(eff_train)
    # eff_train.to_csv('../game/clean_data/eff_train_v1.csv', index=False)
    #
    # eff_test = pd.read_csv('../game/clean_data/eff_test.csv')
    # eff_test = u.trans_time(eff_test)
    # eff_test.to_csv('../game/clean_data/eff_test_v1.csv', index=False)

    # 离散变量编码
    #
    # eff_train_v1 = pd.read_csv('../game/clean_data/eff_train_v1.csv')
    # eff_train_v2 = u.discrete_encode(eff_train_v1)
    # eff_train_v2.to_csv('../game/clean_data/eff_train_v2.csv', index=False)
    #
    # eff_test_v1 = pd.read_csv('../game/clean_data/eff_test_v1.csv')
    # eff_test_v2 = u.discrete_encode(eff_test_v1)
    # eff_test_v2.to_csv('../game/clean_data/eff_test_v2.csv', index=False)

    # 联合贷款有关字段和地理位置
    #
    # eff_train_v2 = pd.read_csv('../game/clean_data/eff_train_v2.csv')
    # eff_train_v3 = u.handle_joint(eff_train_v2)
    # eff_test_v3 = u.trans_geo(eff_train_v3)
    # eff_train_v3.to_csv('../game/clean_data/eff_train_v3.csv', index=False)
    #
    # eff_test_v2 = pd.read_csv('../game/clean_data/eff_test_v2.csv')
    # eff_test_v3 = u.handle_joint(eff_test_v2)
    # eff_test_v3 = u.trans_geo(eff_test_v3)
    # eff_test_v3.to_csv('../game/clean_data/eff_test_v3.csv', index=False)

    # 填充缺失值
    # eff_train_v3 = pd.read_csv('../game/clean_data/eff_train_v3.csv')
    # eff_train_v4 = u.handle_missing(eff_train_v3)
    # eff_train_v4[Columns.rev_rel] = pd.Series(map(int, eff_train_v4[Columns.rev_rel]))
    # print(eff_train_v4.shape[0] - eff_train_v4.count())
    # eff_train_v4.to_csv('../game/clean_data/eff_train_v4.csv', index=False)
    #
    # eff_test_v3 = pd.read_csv('../game/clean_data/eff_test_v3.csv')
    # eff_test_v4 = u.handle_missing(eff_test_v3)
    # eff_train_v4[Columns.rev_rel] = pd.Series(map(int, eff_train_v4[Columns.rev_rel]))
    #
    # eff_test_v4.to_csv('../game/clean_data/eff_test_v4.csv', index=False)

    # 将训练集的正负样例分开
    # eff_train_v4 = pd.read_csv('../game/clean_data/eff_train_v4.csv', low_memory=False)
    # train_pos = eff_train_v4[eff_train_v4[Columns.label] == 1]
    # train_neg = eff_train_v4[eff_train_v4[Columns.label] == 0]
    # train_pos.to_csv('../game/clean_data/train_pos.csv', index=False)
    # train_neg.to_csv('../game/clean_data/train_neg.csv', index=False)

    # eff_train_v4 = pd.read_csv('../game/clean_data/eff_train_v4.csv', low_memory=False)
    # print(eff_train_v4[Columns.total_pymnt_inv].describe())
    # print(eff_train_v4[Columns.total_pymnt].describe())
    # print(eff_train_v4[eff_train_v4[Columns.total_pymnt_inv] != 0][Columns.total_rec_late_fee].describe())






