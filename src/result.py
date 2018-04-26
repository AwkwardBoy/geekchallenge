# -*- coding: utf-8 -*-
from src.sampling import Sampling
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import csv
import os
from collections import Counter


class ResultOut:

    def __init__(self):
        self.s = Sampling

        self.to_predict = pd.read_csv('../game/clean_data/eff_test_v4.csv')
        self.member_ids = self.to_predict.member_id

        self.test_out = '../model/'

    def predict(self):

        predict_set = self.s.data_load('../game/clean_data/eff_test_v4.csv')

        predictions = []

        # load lgb
        for i in range(30):
            file = '../model/lgb_model/model_' + str(i) + '.lgb'
            lgb_classifer = lgb.Booster(model_file=file)
            pred = lgb_classifer.predict(predict_set)
            predictions.append(pred)

        # load xgb
        predict_set = xgb.DMatrix(predict_set)
        for j in range(30):
            xgb_classifier = xgb.Booster()
            xgb_classifier.load_model('../model/xgb_model/model_' + str(j) + '.xgb')
            pred = xgb_classifier.predict(predict_set)
            predictions.append(pred)

        # load big lgb_model
        # for i in range(30):
        #     file = '../model/lgb_model/big_model_' + str(i) + '.lgb'
        #     lgb_classifer = lgb.Booster(model_file=file)
        #     pred = lgb_classifer.predict(predict_set)
        #     predictions.append(pred)

        predictions = np.array(predictions)
        predictions = np.mean(predictions, axis=0)
        print(predictions)
        return predictions

    def csv_out(self, preds):

        preds = np.array(list(map(lambda x: int(x >= 0.86), preds)))
        with(open(os.path.join(self.test_out, "test.csv"), mode="w")) as outer:
            writer = csv.writer(outer)
            writer.writerow(["member_id", "acc_now_delin"])
            for pairs in list(zip(self.member_ids, preds)):
                writer.writerow([pairs[0], pairs[1]])

    @staticmethod
    def check_out():
        result = pd.read_csv('../model/test.csv')
        print(Counter(result.acc_now_delin))


if __name__ == '__main__':
    ro = ResultOut()
    p = ro.predict()
    ro.csv_out(p)
    ro.check_out()
