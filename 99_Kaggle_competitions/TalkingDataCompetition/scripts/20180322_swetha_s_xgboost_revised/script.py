import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import gc

path = '../../input/'
traincolumns = ['ip','app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8'
        }
 
train = pd.read_csv(path+'train.csv', skiprows=range(1,144903891), nrows=40000000, 
                    usecols = traincolumns, dtype=dtypes)
test = pd.read_csv(path+'test.csv', dtype=dtypes)
sub = pd.DataFrame()
sub['click_id'] = test['click_id']
train = train.dropna()

def preprocessClicktime(df):
	# Make some new features with click_time column
	df["datetime"] = pd.to_datetime(df['click_time'])
	df['dow']      = df['datetime'].dt.dayofweek.astype('uint8')
	df['hour'] 	   = df['datetime'].dt.hour.astype('uint8')
	df['minute']   = df['datetime'].dt.minute.astype('uint8')
	df['second']   = df['datetime'].dt.second.astype('uint8')
	return df

train = preprocessClicktime(train)
train = train.drop(['click_time','datetime'],axis=1)
gc.collect()

test = preprocessClicktime(test)
test = test.drop(['click_id','click_time','datetime'],axis=1)
gc.collect()

y = train['is_attributed']
train = train.drop(['is_attributed'], axis=1)

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
merge['clicks_by_ip'] = merge['clicks_by_ip'].astype('int32')
del ip_count

train = merge[:nrow_train]
test = merge[nrow_train:]
del merge
gc.collect()

params = {'eta': 0.15, # learning rate
          'tree_method': "auto", 
          'max_depth': 4, 
          'subsample': 0.8, 
          'colsample_bytree': 0.7, 
          'colsample_bylevel':0.7,
          'min_child_weight':0,
          'alpha':4,
          'objective': 'binary:logistic', 
          'scale_pos_weight':9,
          'eval_metric': 'auc', 
          'random_state': 99,
 #         'threads': 5,
          'silent': True}
          
x1, x2, y1, y2 = train_test_split(train, y, test_size=0.1, shuffle=False)
del train
del y
gc.collect()

dtrain = xgb.DMatrix(x1, y1)
del x1, y1
gc.collect()

dvalid = xgb.DMatrix(x2, y2)
del x2, y2
gc.collect()

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
model = xgb.train(params, dtrain, 1000, watchlist, maximize=True, early_stopping_rounds=70, verbose_eval=10)
del dtrain
del dvalid
gc.collect()

dtest = xgb.DMatrix(test)
del test
gc.collect()

sub['is_attributed'] = model.predict(dtest, ntree_limit=model.best_ntree_limit)
sub.to_csv('../../output/xgb_sub5.csv',index=False)