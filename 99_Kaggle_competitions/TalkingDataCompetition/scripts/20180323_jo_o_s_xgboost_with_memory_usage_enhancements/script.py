# The original is
# https://www.kaggle.com/joaopmpeinado/xgboost-with-new-features-lb-0-961?scriptVersionId=2790617
# (of which there are newer versions available)
# from João Pedro Peinado

# This version was revised by Andy Halress to improve memory usage, 
# so as allow it to process a larger subset of the training data.
# Also, I changed the validation split to be timewise instead of random.

# João's earlier improvement is based on a kernel from Pranav Pandya:
# https://www.kaggle.com/pranav84/xgboost-on-hist-mode-ip-addresses-dropped

import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import gc

# Change this for validation with 10% from train
using_test = True

path = '../../input/'

def dataPreProcessTime(df):
    # Make some new features with click_time column
    df['datetime'] = pd.to_datetime(df['click_time'])
    df['dow']      = df['datetime'].dt.dayofweek
    df.drop(['click_time', 'datetime'], axis=1, inplace=True)
    return df

start_time = time.time()

columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        }

# Read the last lines because they are more impacting in training than the starting lines
train = pd.read_csv(path+"train.csv", skiprows=range(1,129903891), nrows=55000000, usecols=columns, dtype=dtypes)
test = pd.read_csv(path+"test.csv")

print('[{}] Finished to load data'.format(time.time() - start_time))

train = dataPreProcessTime(train)
test = dataPreProcessTime(test)
gc.collect()

# Drop the IP and the columns from target
y = train['is_attributed']
train.drop(['is_attributed'], axis=1, inplace=True)

# Drop IP and ID from test rows
sub = pd.DataFrame()
sub['click_id'] = test['click_id']
test.drop(['click_id'], axis=1, inplace=True)
gc.collect()

# Some feature engineering
nrow_train = train.shape[0]
merge = pd.concat([train, test])
del train, test
gc.collect()

# Count the number of clicks by ip
ip_count = merge.groupby('ip')['app'].count().reset_index()
ip_count.columns = ['ip', 'clicks_by_ip']
merge = pd.merge(merge, ip_count, on='ip', how='left', sort=False)
merge.drop('ip', axis=1, inplace=True)

train = merge[:nrow_train]
test = merge[nrow_train:]

del ip_count
del merge
gc.collect()

print('[{}] Start XGBoost Training'.format(time.time() - start_time))

# Set the params(this params from Pranav kernel) for xgboost model
params = {'eta': 0.6,
          'tree_method': "hist",
          'grow_policy': "lossguide",
          'max_leaves': 1400,  
          'max_depth': 0, 
          'subsample': 0.9, 
          'colsample_bytree': 0.7, 
          'colsample_bylevel':0.7,
          'min_child_weight':0,
          'alpha':4,
          'objective': 'binary:logistic', 
          'scale_pos_weight':9,
          'eval_metric': 'auc', 
          'nthread':8,
          'random_state': 99, 
          'silent': True}

if (using_test == False):
    # Get 10% of train dataset to use as validation
    x1, x2, y1, y2 = train_test_split(train, y, test_size=0.1, shuffle=False)
    del train, y
    dtrain = xgb.DMatrix(x1, y1)
    del x1, y1
    dvalid = xgb.DMatrix(x2, y2)
    del x2, y2 
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    earlystop = 20
    nrounds = 300
    verbose = 2
else:
    dtrain = xgb.DMatrix(train, y)
    del train, y
    watchlist = [(dtrain, 'train')]
    verbose = 1
    earlystop = None
    nrounds = 14
gc.collect()

dtest = xgb.DMatrix(test)
del test
gc.collect()

model = xgb.train(params, dtrain, nrounds, watchlist, maximize=True, 
                  early_stopping_rounds=earlystop, verbose_eval=verbose)

del dtrain

print('[{}] Finish XGBoost Training'.format(time.time() - start_time))

sub['is_attributed'] = model.predict(dtest, ntree_limit=model.best_ntree_limit)
sub.to_csv('../../output/xgb_sub.csv',index=False)