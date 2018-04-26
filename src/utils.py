# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from collections import Counter
import datetime
from nltk import stem
import geocoder


class Columns:
    member_id = 'member_id'
    loan_amnt = 'loan_amnt'
    funded_amnt = 'funded_amnt'
    funded_amnt_inv = 'funded_amnt_inv'
    term = 'term'
    int_rate = 'int_rate'
    installment = 'installment'
    grade = 'grade'
    sub_grade = 'sub_grade'
    emp_title = 'emp_title'
    emp_length = 'emp_length'
    home_ownership = 'home_ownership'
    annual_inc = 'annual_inc'
    verification_status = 'verification_status'
    issue_d = 'issue_d'
    loan_status = 'loan_status'
    pymnt_plan = 'pymnt_plan'
    desc = 'desc'
    purpose = 'purpose'
    title = 'title'
    zip_code = 'zip_code'
    addr_state = 'addr_state'
    dti = 'dti'
    earliest_cr_line = 'earliest_cr_line'
    mths_since_last_record = 'mths_since_last_record'
    pub_rec = 'pub_rec'
    revol_bal = 'revol_bal'
    revol_util = 'revol_util'
    total_acc = 'total_acc'
    initial_list_status = 'initial_list_status'
    out_prncp = 'out_prncp'
    out_prncp_inv = 'out_prncp_inv'
    total_pymnt = 'total_pymnt'
    total_pymnt_inv = 'total_pymnt_inv'
    total_rec_prncp = 'total_rec_prncp'
    total_rec_int = 'total_rec_int'
    total_rec_late_fee = 'total_rec_late_fee'
    recoveries = 'recoveries'
    collection_recovery_fee = 'collection_recovery_fee'
    collections_12_mths_ex_med = 'collections_12_mths_ex_med'
    mths_since_last_major_derog = 'mths_since_last_major_derog'
    policy_code = 'policy_code'
    application_type = 'application_type'
    annual_inc_joint = 'annual_inc_joint'
    dti_joint = 'dti_joint'
    verification_status_joint = 'verification_status_joint'
    acc_now_delinq = 'acc_now_delinq'
    tot_coll_amt = 'tot_coll_amt'
    tot_cur_bal = 'tot_cur_bal'
    open_acc_6m = 'open_acc_6m'
    open_il_6m = 'open_il_6m'
    open_il_12m = 'open_il_12m'
    open_il_24m = 'open_il_24m'
    mths_since_rcnt_il = 'mths_since_rcnt_il'
    total_bal_il = 'total_bal_il'
    il_util = 'il_util'
    open_rv_12m = 'open_rv_12m'
    open_rv_24m = 'open_rv_24m'
    max_bal_bc = 'max_bal_bc'
    all_util = 'all_util'
    rev_rel = "rev_rel"
    total_rev_hi_lim = 'total_rev_hi_lim'
    inq_fi = 'inq_fi'
    total_cu_tl = 'total_cu_tl'
    inq_last_12m = 'inq_last_12m'
    lat = 'lat'
    lng = 'lng'
    avg_loan = 'avg_loan'

    discrete_columns = [
        term, title, home_ownership, initial_list_status, loan_status, verification_status,
        zip_code, pymnt_plan, rev_rel, application_type, verification_status_joint, addr_state
    ]

    continuous_columns = [
        loan_amnt, funded_amnt, funded_amnt_inv, int_rate, installment,
        annual_inc, mths_since_last_major_derog, pub_rec, revol_bal, revol_util,
        total_acc, out_prncp, out_prncp_inv, total_pymnt, total_pymnt_inv,
        total_rec_prncp, total_rec_int, total_rec_late_fee, recoveries, collection_recovery_fee,
        collections_12_mths_ex_med, tot_coll_amt, tot_cur_bal, total_rev_hi_lim, annual_inc_joint,
        dti_joint, purpose
    ]

    label = acc_now_delinq
    id = member_id

    features = [
        loan_amnt, funded_amnt, funded_amnt_inv, term, int_rate,
        installment, grade, sub_grade, emp_length, home_ownership,
        annual_inc, verification_status, issue_d, loan_status, pymnt_plan,
        addr_state, dti, earliest_cr_line, pub_rec, verification_status_joint,
        revol_bal, revol_util, total_acc, initial_list_status, out_prncp,
        out_prncp_inv, total_pymnt, total_pymnt_inv, total_rec_prncp, total_rec_int,
        total_rec_late_fee, recoveries, collection_recovery_fee, collections_12_mths_ex_med, policy_code,
        application_type, tot_coll_amt, tot_cur_bal, total_rev_hi_lim, annual_inc_joint,
        dti_joint, purpose
    ]

    out_features = [
        all_util, il_util, inq_fi, inq_last_12m, emp_title, desc,
        max_bal_bc, open_il_12m, mths_since_rcnt_il, open_acc_6m, open_il_6m, mths_since_last_record,
        open_il_24m, open_rv_12m, open_rv_24m, total_bal_il, total_cu_tl, mths_since_last_major_derog
    ]


'''
    样本比例
    {0: 706610, 1: 3293}
    
'''


class Utils:

    def __init__(self):
        pass

    def trans_time(self, df):
        df[Columns.earliest_cr_line] = pd.Series(map(self.transfer_time, df[Columns.earliest_cr_line]))
        df[Columns.issue_d] = pd.Series(map(self.transfer_time, df[Columns.issue_d]))
        df[Columns.emp_length] = pd.Series(map(self.trans_years, df[Columns.emp_length]))
        return df

    # 样本缺失值
    @staticmethod
    def handle_missing(df):

        # 初步删除缺失值超90% 的数据行
        rows = df.shape[0]
        for column in df.columns:
            if sum(df[column].isna()) / rows > 0.75:
                df.drop(columns=column)

        # todo 需要填充的列

        # done annual_inc: 4 float
        # done collections_12_mths_ex_med: 118 int
        # done earliest_cr_line: 24  datetime
        # done emp_length: 35852 int
        # todo emp_title: 41187 str
        # done revol_util: 409 float
        # done title: 128 str
        # done tot_coll_amt: 56135
        # done tot_cur_bal: float 56135
        # done tot_acc: 24 float 24
        # done total_rev_hi_lim: 56135

        df.loc[df[Columns.earliest_cr_line].isna(), Columns.earliest_cr_line] = \
            df[Columns.earliest_cr_line].sum() // df[df[Columns.earliest_cr_line].notna()].shape[0]

        df.loc[df[Columns.annual_inc].isna(), Columns.annual_inc] = \
            df[Columns.annual_inc].sum() // df[df[Columns.annual_inc].notna()].shape[0]

        df.loc[df[Columns.emp_length].isna(), Columns.emp_length] = \
            df[Columns.emp_length].sum() // df[df[Columns.emp_length].notna()].shape[0]

        df.loc[df[Columns.pub_rec].isna(), Columns.pub_rec] = \
            np.median(df[df[Columns.pub_rec].notna()][Columns.pub_rec])

        df.loc[df[Columns.collections_12_mths_ex_med].isna(), Columns.collections_12_mths_ex_med] = \
            df[Columns.collections_12_mths_ex_med].sum() // df[df[Columns.collections_12_mths_ex_med].notna()].shape[0]

        df.loc[df[Columns.revol_util].isna(), Columns.revol_util] = \
            df[Columns.revol_util].sum() // df[df[Columns.revol_util].notna()].shape[0]

        df.loc[df[Columns.total_acc].isna(), Columns.total_acc] = \
            df[Columns.total_acc].sum() // df[df[Columns.total_acc].notna()].shape[0]

        df.loc[df[Columns.dti_joint].isna(), Columns.dti_joint] = \
            df[Columns.dti_joint].sum() // df[df[Columns.dti_joint].notna()].shape[0]

        df[Columns.rev_rel] = pd.Series(map(int, df[Columns.total_rev_hi_lim].isna()))

        df.loc[df[Columns.total_rev_hi_lim].isna(), Columns.total_rev_hi_lim] = \
            df[Columns.total_rev_hi_lim].sum() // df[df[Columns.total_rev_hi_lim].notna()].shape[0]

        df.loc[df[Columns.tot_cur_bal].isna(), Columns.tot_cur_bal] = \
            df[Columns.tot_cur_bal].sum() // df[df[Columns.tot_cur_bal].notna()].shape[0]

        df.loc[df[Columns.tot_coll_amt].isna(), Columns.tot_coll_amt] = \
            df[Columns.tot_coll_amt].sum() // df[df[Columns.tot_coll_amt].notna()].shape[0]

        return df

    # 离散数据编码
    # done term, home_ownership, initial_list_status, loan_status, pymnt_plan,
    # done addr_state, grade, sub_grade, purpose

    def discrete_encode(self, df):

        df[Columns.home_ownership] = pd.Series(map(self.encode_owner, df[Columns.home_ownership]))
        df[Columns.term] = pd.Series(map(self.encode_term, df[Columns.term]))
        df[Columns.initial_list_status] = pd.Series(map(self.encode_init_status, df[Columns.initial_list_status]))
        df[Columns.loan_status] = pd.Series(map(self.encode_loan_status, df[Columns.loan_status]))
        df[Columns.pymnt_plan] = pd.Series(map(self.encode_plan, df[Columns.pymnt_plan]))
        df[Columns.sub_grade] = pd.Series(map(self.encode_grade, df[Columns.sub_grade]))
        df[Columns.grade] = pd.Series(map(int, df[Columns.sub_grade]))
        df[Columns.purpose] = pd.Series(map(self.encode_purpose, df[Columns.purpose]))

        return df

    # home_ownership
    @staticmethod
    def encode_owner(owner_ship):
        owner_ship_dict = {
            'MORTGAGE': 1,
            'RENT': 2,
            'OWN': 3,
            'NONE': 4,
            'ANY': 5,
            'OTHER': 6
        }
        try:
            return owner_ship_dict[owner_ship]
        except KeyError:
            pass

    # term
    @staticmethod
    def encode_term(term):
        terms = {
            ' 36 months': 1,
            ' 60 months': 2
        }
        try:
            return terms[term]
        except KeyError:
            pass

    # init_status
    @staticmethod
    def encode_init_status(status):
        init_status = {
            'f': 1,
            'w': 2
        }
        try:
            return init_status[status]
        except KeyError:
            pass

    # loan status
    @staticmethod
    def encode_loan_status(loan_status):

        status = {
            'Current': 1,
            'Fully Paid': 2,
            'Charged Off': 3,
            'Late (16-30 days)': 4,
            'Late (31-120 days)': 5,
            'Issued': 6,
            'In Grace Period': 7,
            'Does not meet the credit policy. Status:Fully Paid': 8,
            'Does not meet the credit policy. Status:Charged Off': 9,
            'Default': 10
        }
        try:
            return status[loan_status]
        except KeyError:
            pass

    # pymnt_plan
    @staticmethod
    def encode_plan(if_plan):
        plans = {
            'n': 1,
            'y': 2
        }
        try:
            return plans[if_plan]
        except KeyError:
            pass

    # sub_grade encode
    @staticmethod
    def encode_grade(sub):

        main_grade = sub[0]
        level = int(sub[1])
        grade_dict = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}

        main_grade = grade_dict[main_grade]
        sub_grade = (3 - level) * 0.2
        try:
            return main_grade + sub_grade
        except KeyError:
            pass

    @staticmethod
    def transfer_time(time_stamp):
        try:
            t = datetime.datetime.strptime(time_stamp, "%b-%Y")
            now = datetime.datetime(year=2018, month=4, day=1)
            months = (now - t).days // 30
            return months
        except TypeError:
            pass

    @staticmethod
    def trans_years(emp_length):
        trans_dict = {'< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
                      '4 years': 4, '5 years': 5, '6 years': 6,
                      '7 years': 7, '8 years': 8, '9 years': 9, '10+ years': 10}
        try:
            years = trans_dict[emp_length]
            return years
        except KeyError:
            pass

    @staticmethod
    def encode_purpose(purpose):
        purpose_dict = {
            'debt_consolidation': 1,
            'credit_card': 2,
            'home_improvement': 3,
            'major_purchase': 4,
            'small_business': 5,
            'car': 5,
            'medical': 6,
            'moving': 7,
            'vacation': 8,
            'house': 9,
            'wedding': 10,
            'renewable_energy': 11,
            'educational': 12,
            'other': 13
        }
        try:
            return purpose_dict[purpose]
        except KeyError:
            pass

    @staticmethod
    def trans_geo(df):
        addresses = df[Columns.addr_state]
        addr_dict = {
            'NV': [38.8026097, -116.419389],
            'UT': [30.2849185, -97.7340567],
            'RI': [41.5800945, -71.4774291],
            'PA': [41.2033216, -77.1945247],
            'OH': [40.4172871, -82.90712300000001],
            'NM': [34.5199402, -105.8700901],
            'ND': [47.5514926, -101.0020119],
            'NH': [43.1938516, -71.5723953],
            'IA': [41.8780025, -93.097702],
            'MA': [42.4072107, -71.3824374],
            'TX': [31.9685988, -99.9018131],
            'VT': [37.22838429999999, -80.42341669999999], 
            'WA': [47.7510741, -120.7401385],
            'CA': [36.778261, -119.4179324],
            'CO': [37.0301138, -95.61760350000002],
            'DC': [38.9071923, -77.0368707],
            'MT': [46.8796822, -110.3625658],
            'AK': [64.2008413, -149.4936733],
            'IN': [40.2671941, -86.1349019],
            'DE': [37.687622, -95.4653173],
            'HI': [37.6880733, -95.46100940000001],
            'VA': [37.4315734, -78.6568942],
            'NC': [35.7595731, -79.01929969999999],
            'TN': [35.5174913, -86.5804473],
            'ID': [37.0657963, -95.7777715],
            'LA': [34.0522342, -118.2436849],
            'SD': [43.9695148, -99.9018131],
            'ME': [36.2046771, -95.95761270000001],
            'OK': [35.0077519, -97.092877],
            'AZ': [34.0489281, -111.0937311],
            'OR': [43.8041334, -120.5542012],
            'AR': [35.20105, -91.8318334],
            'FL': [27.6648274, -81.5157535],
            'KY': [37.8393332, -84.2700179],
            'MS': [37.09024, -95.712891],
            'GA': [32.1656221, -82.9000751],
            'MI': [44.3148443, -85.60236429999999],
            'SC': [33.836081, -81.1637245],
            'MO': [37.9642529, -91.8318334],
            'CT': [41.6032207, -73.087749],
            'NY': [40.7127753, -74.0059728],
            'KS': [39.011902, -98.4842465],
            'WY': [43.0759678, -107.2902839],
            'MD': [39.0457549, -76.64127119999999],
            'MN': [46.729553, -94.6858998],
            'WI': [43.7844397, -88.7878678],
            'AL': [32.3182314, -86.902298],
            'NE': [41.4925374, -99.9018131],
            'NJ': [40.0583238, -74.4056612],
            'IL': [40.6331249, -89.3985283],
            'WV': [38.5976262, -80.4549026],
        }
        lats = []
        lngs = []
        for address in addresses:
            lats.append(addr_dict[address][0])
            lngs.append(addr_dict[address][1])

        df['lat'] = pd.Series(lats)
        df['lng'] = pd.Series(lngs)
        df.drop(columns=[Columns.addr_state])
        return df

    def handle_joint(self, df):
        df[Columns.annual_inc_joint] = df[Columns.annual_inc_joint] - df[Columns.annual_inc]

        df.loc[df[Columns.application_type] == 'INDIVIDUAL', Columns.annual_inc_joint] = 0
        df[Columns.dti_joint] = df[Columns.dti_joint] - df[Columns.dti]

        df.loc[df[Columns.application_type] == 'INDIVIDUAL', Columns.dti_joint] = 0

        df.loc[df[Columns.application_type] == 'INDIVIDUAL', Columns.verification_status_joint] = 'no'
        df[Columns.verification_status] = pd.Series(map(self.encode_verified, df[Columns.verification_status]))
        df[Columns.verification_status_joint] = \
            pd.Series(map(self.encode_verified, df[Columns.verification_status_joint]))

        df[Columns.application_type] = pd.Series(map(self.encode_application, df[Columns.application_type]))

        return df

    @staticmethod
    def encode_application(app_type):
        app_types = {
            'JOINT': 1,
            'INDIVIDUAL': 2
        }
        try:
            return app_types[app_type]
        except KeyError:
            pass
           
    @staticmethod
    def encode_verified(verified_type):
        verified_types = {
            'Source Verified': 1,
            'Verified': 1,
            'Not Verified': 2,
            'no': 3
        }
        try:
            return verified_types[verified_type]
        except KeyError:
            pass

    # 检查一些类别特征
    @staticmethod
    def count_na(df):
        return Counter(df.count(axis=1))
    # train_set
    # {44: 341703, 45: 213984, 42: 48534, 46: 48278, 41: 18476, 43: 17723, 58: 8951, 59: 4807,
    #  47: 1817, 40: 1541, 57: 1387, 60: 954, 56: 719, 39: 607, 55: 152, 48: 80, 54: 48, 61: 47, 62: 45,
    #  37: 13, 49: 13, 63: 10, 36: 7, 35: 4, 38: 2, 53: 1}
    # test_set
    # {43: 85776, 44: 53168, 41: 12256, 45: 12032, 40: 4608, 42: 4382, 57: 2182, 58: 1229, 46: 451,
    #  56: 393, 39: 376, 59: 220, 55: 150, 38: 143, 54: 35, 47: 25, 53: 16, 60: 16, 61: 9, 36: 3, 48: 3, 35: 2, 62: 1}

    @staticmethod
    def check_object(df):
        pass
        # verification_status
        # print(Counter(df[Columns.verification_status]))
        # {'Source Verified': 263587, 'Verified': 232859, 'Not Verified': 213457}

        # application_type
        # print(Counter(df[Columns.application_type]))
        # {'INDIVIDUAL': 709493, 'JOINT': 410}

    @staticmethod
    def check_add_state(df):
        address = list(df[Columns.addr_state])
        label = list(df[Columns.label])
        stats_info = {}

        address_info = Counter(list(zip(address, label)))
        for k, v in address_info.items():
            if k[0] not in stats_info:
                stats_info[k[0]] = {}
                stats_info[k[0]][k[1]] = v
            else:
                stats_info[k[0]][k[1]] = v
        for add, count_info in stats_info.items():
            print(count_info)
            try:
                count_info['ratio'] = count_info[1] / count_info[0]
            except KeyError:
                count_info['ratio'] = 0
        return stats_info

    # {'MN': {0: 12706, 1: 47, 'ratio': 0.003699039823705336}, 'SD': {0: 1447, 1: 7, 'ratio': 0.0048375950241879755},
    # 'NM': {0: 3929, 1: 20, 'ratio': 0.0050903537795876815}, 'ME': {0: 407, 1: 1, 'ratio': 0.002457002457002457},
    # 'TX': {0: 56554, 1: 326, 'ratio': 0.005764402164303144}, 'OH': {0: 23573, 1: 114, 'ratio': 0.004836041233614728},
    # 'AR': {0: 5315, 1: 29, 'ratio': 0.005456255879586077}, 'CT': {0: 10793, 1: 68, 'ratio': 0.006300379875845455},
    # 'WV': {0: 3468, 1: 23, 'ratio': 0.0066320645905421}, 'NC': {0: 19658, 1: 101, 'ratio': 0.005137857360870892},
    # 'TN': {0: 10282, 1: 57, 'ratio': 0.0055436685469752965}, 'HI': {0: 3676, 1: 14, 'ratio': 0.003808487486398259},
    # 'DC': {0: 1951, 1: 5, 'ratio': 0.0025627883136852894}, 'AL': {0: 8849, 1: 45, 'ratio': 0.005085320375183637},
    # 'SC': {0: 8483, 1: 40, 'ratio': 0.004715312978898974}, 'MT': {0: 2050, 1: 10, 'ratio': 0.004878048780487805},
    # 'FL': {0: 48458, 1: 215, 'ratio': 0.004436831895662223}, 'KY': {0: 6815, 1: 33, 'ratio': 0.004842259721203228},
    # 'ID': {0: 12, 'ratio': 0}, 'RI': {0: 3074, 1: 21, 'ratio': 0.0068314899154196486}, 'IA': {0: 13, 'ratio': 0},
    # 'VT': {0: 1427, 1: 10, 'ratio': 0.00700770847932726}, 'VA': {0: 20981, 1: 71, 'ratio': 0.0033840141080024783},
    # 'MS': {0: 3058, 1: 19, 'ratio': 0.006213211249182472}, 'OK': {0: 6437, 1: 31, 'ratio': 0.004815908031691782},
    # 'CO': {0: 14964, 1: 53, 'ratio': 0.003541833734295643}, 'PA': {0: 25091, 1: 109, 'ratio': 0.0043441871587421785},
    # 'CA': {0: 102980, 1: 398, 'ratio': 0.00386482812196543}, 'OR': {0: 8698, 1: 37, 'ratio': 0.0042538514601057715},
    # 'IL': {0: 28250, 1: 113, 'ratio': 0.004}, 'WY': {0: 1599, 1: 12, 'ratio': 0.0075046904315197},
    # 'NY': {0: 58955, 1: 310, 'ratio': 0.005258247816130947}, 'WA': {0: 15428, 1: 56, 'ratio': 0.003629764065335753},
    # 'NJ': {0: 26576, 1: 166, 'ratio': 0.006246237206502107}, 'WI': {0: 9257, 1: 25, 'ratio': 0.002700658960786432},
    # 'LA': {0: 8451, 1: 42, 'ratio': 0.004969826056088037}, 'MA': {0: 16441, 1: 84, 'ratio': 0.005109178273827626},
    # 'MO': {0: 11308, 1: 45, 'ratio': 0.003979483551467987}, 'MI': {0: 18311, 1: 101, 'ratio': 0.005515810168751024},
    # 'ND': {0: 396, 1: 4, 'ratio': 0.010101010101010102}, 'IN': {0: 10971, 1: 64, 'ratio': 0.005833561206817974},
    # 'AK': {0: 1778, 1: 2, 'ratio': 0.0011248593925759281}, 'AZ': {0: 16343, 1: 68, 'ratio': 0.0041608027901854005},
    # 'NH': {0: 3396, 1: 8, 'ratio': 0.002355712603062426}, 'NE': {0: 921, 1: 7, 'ratio': 0.00760043431053203},
    # 'GA': {0: 23007, 1: 106, 'ratio': 0.004607293432433607}, 'DE': {0: 1993, 1: 8, 'ratio': 0.004014049172102358},
    # 'NV': {0: 9997, 1: 53, 'ratio': 0.005301590477143143}, 'MD': {0: 16730, 1: 76, 'ratio': 0.004542737597130903},
    # 'UT': {0: 4968, 1: 16, 'ratio': 0.00322061191626409}, 'KS': {0: 6385, 1: 23, 'ratio': 0.003602192638997651}}

    @staticmethod
    def check_title(df):
        s = stem.SnowballStemmer('english')
        all_stems = []
        for title in df[Columns.title]:
            title = str(title).strip().lower().split()
            title = [s.stem(word) for word in title]
            all_stems.extend(title)
        print(sorted(Counter(all_stems).items(), key=lambda a: a[1], reverse=True)[0:100])

    @staticmethod
    def check_emp_title(df):
        s = stem.SnowballStemmer('english')
        all_stems = []
        for title in df[Columns.emp_title]:
            title = str(title).strip().lower().split()
            title = [s.stem(word) for word in title]
            all_stems.extend(title)
        print(sorted(Counter(all_stems).items(), key=lambda a: a[1], reverse=True)[0:100])


if __name__ == '__main__':
    u = Utils()
    train = pd.read_csv('../game/train.csv', low_memory=False)
    print(train.shape[0] - train.count())
    # print(set(train[Columns.verification_status_joint]))
    # print(set(train[Columns.verification_status]))
    # print(set(train[Columns.application_type]))
    # print(train[Columns.annual_inc_joint].count())
    # print((train[Columns.annual_inc_joint] - train[Columns.annual_inc]).count())
    # print(Counter(train[Columns.home_ownership]))
    # print(Counter(train[Columns.term]))
    # print(Counter(train[Columns.initial_list_status]))
    # print(Counter(train[Columns.loan_status]))
    # print(Counter(train[Columns.pymnt_plan]))
    # print(train.shape[0] - train.count())
    # print(train[Columns.verification_status])
    # test = pd.read_csv('./game/test.csv', low_memory=False)

    # print(sorted(Counter(train[Columns.title]).items(), key=lambda a: a[1], reverse=True)[0:100])

    # u.check_object(train)


