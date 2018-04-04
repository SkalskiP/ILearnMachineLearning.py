# num_leaves :  7  ->  9
# max_depth  :  4  ->  5
# subsample  : 0.7 -> 0.9
# scale_pos_weight : 99.7 -> 300
# validate on selected hr from last day
# new 'device_app', 'device_hr', 'channel_hr'

import pandas as pd
import time
import gc
import numpy as np
from sklearn.model_selection import train_test_split 
import lightgbm as lgb
from datetime import datetime
import time
start = time.time()

# Constants and general
MAX_ROUNDS = 1000
EARLY_STOP = 50
OPT_ROUNDS = 680

# Paths and file names
outfile = 'sub_lgbm_r_to_python_withcv.csv'
path = '../../input/'

# Columns types
dtypes = {
    'ip'            : 'uint32',
    'app'           : 'uint16',
    'device'        : 'uint16',
    'os'            : 'uint16',
    'channel'       : 'uint16',
    'is_attributed' : 'uint8',
    'click_id'      : 'uint32'
}

# Columns names
train_cols = ['ip','app','device','os', 'channel', 'click_time', 'is_attributed']
test_cols = ['ip','app','device','os', 'channel', 'click_time', 'click_id']

target = 'is_attributed'
predictors = ['app',
              'device',
              'os',
              'channel',
              'hour',
              'ip_hr',
              'ip_hr_app',
              'ip_hr_os',
              'ip_hr_dev',
              'ip_min',
              'ip_min_app',
              'ip_min_os',
              'device_app',
              'device_hr',
              'channel_hr']

categorical = ['app', 'device', 'os', 'channel', 'hour']


# Limit dates for the training and test set
train_start_date = "2017-11-08 00:00:00"
train_end_date = "2017-11-08 23:59:59"
valid_start_date = "2017-11-09 04:00:00"
valid_end_date = "2017-11-09 16:00:00"

train_start_date = datetime.strptime(train_start_date, '%Y-%m-%d %H:%M:%S')
train_end_date = datetime.strptime(train_end_date, '%Y-%m-%d %H:%M:%S')
valid_start_date = datetime.strptime(valid_start_date, '%Y-%m-%d %H:%M:%S')
valid_end_date = datetime.strptime(valid_end_date, '%Y-%m-%d %H:%M:%S')

# Filtration by date
def filtrationByDateTrain(df):
    print("Converting to datetime...")
    df['click_time'] = pd.to_datetime(df['click_time'])
    print("Filtration of dataset...")
    return df[(df['click_time'] <= train_end_date) & (df['click_time'] > train_start_date)]

#---------------------------------------------------------------------------------

def filtrationByDateValid(df):
    print("Converting to datetime...")
    df['click_time'] = pd.to_datetime(df['click_time'])
    print("Filtration of dataset...")
    return df[(df['click_time'] <= valid_end_date) & (df['click_time'] > valid_start_date)]

#---------------------------------------------------------------------------------

def prep_data( df ):
    print('Prep data...')

    print("Extract day and hour from datetime...")
    df['hour'] = df['click_time'].dt.hour.astype('uint8')
    df['day'] = df['click_time'].dt.dayofyear.astype('uint8')
    df['minute'] = df['click_time'].dt.minute.astype('uint8')
    df['second'] = df['click_time'].dt.second.astype('uint8')
    df.drop(['click_time'], axis=1, inplace=True)
    gc.collect()
    
    print("Aggregation of features inside hour...")
    
    print("Group by ip, day, hour...")
    gp = df[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day',
             'hour'])[['channel']].count().reset_index().rename(index=str, 
             columns={'channel': 'ip_hr'})
    df = df.merge(gp, on=['ip','day','hour'], how='left')
    df['ip_hr'] = df['ip_hr'].astype('uint32')
    print( "ip_hr max val = ", df.ip_hr.max() )
    del gp
    gc.collect()
    
    print("Group by ip, day, hour, app...")
    gp = df[['ip', 'day', 'hour', 'app', 'channel']].groupby(by=['ip', 'day',
             'hour', 'app'])[['channel']].count().reset_index().rename(index=str, 
             columns={'channel': 'ip_hr_app'})
    df = df.merge(gp, on=['ip','day','hour','app'], how='left')
    df['ip_hr_app'] = df['ip_hr_app'].astype('uint32')
    print( "ip_hr_app max val = ", df.ip_hr_app.max() )
    del gp
    gc.collect()
    
    print("Group by ip, day, hour, os...")
    gp = df[['ip', 'day', 'hour', 'os', 'channel']].groupby(by=['ip', 'day',
             'hour', 'os'])[['channel']].count().reset_index().rename(index=str, 
             columns={'channel': 'ip_hr_os'})
    df = df.merge(gp, on=['ip','day','hour','os'], how='left')
    df['ip_hr_os'] = df['ip_hr_os'].astype('uint32')
    print( "ip_hr_os max val = ", df.ip_hr_os.max() )
    del gp
    gc.collect()
    
    print("Group by ip, day, hour, device...")
    gp = df[['ip', 'day', 'hour', 'device', 'channel']].groupby(by=['ip', 'day',
             'hour', 'device'])[['channel']].count().reset_index().rename(index=str, 
             columns={'channel': 'ip_hr_dev'})
    df = df.merge(gp, on=['ip','day','hour','device'], how='left')
    df['ip_hr_dev'] = df['ip_hr_dev'].astype('uint32')
    print( "ip_hr_dev max val = ", df.ip_hr_dev.max() )
    del gp
    gc.collect()

    print("Group by device, hour...")
    gp = df[['device', 'hour', 'channel']].groupby(by=['device', 
            'hour'])[['channel']].count().reset_index().rename(index=str, 
             columns={'channel': 'device_hr'})
    df = df.merge(gp, on=['device', 'hour'], how='left')
    df['device_hr'] = df['device_hr'].astype('uint32')
    print( "device_hr max val = ", df.device_hr.max() )
    del gp
    gc.collect()

    print("Group by channel, hour...")
    gp = df[['ip', 'hour', 'channel']].groupby(by=['channel', 
            'hour'])[['ip']].count().reset_index().rename(index=str, 
             columns={'channel': 'channel_hr'})
    df = df.merge(gp, on=['channel', 'hour'], how='left')
    df['channel_hr'] = df['channel_hr'].astype('uint32')
    print( "channel_hr max val = ", df.channel_hr.max() )
    del gp
    gc.collect()
    
    print("Aggregation of features inside minute...")
    
    print("Group by ip, day, hour, minute...")
    gp = df[['ip', 'day', 'hour', 'minute', 'channel']].groupby(by=['ip', 'day',
             'hour','minute'])[['channel']].count().reset_index().rename(index=str, 
             columns={'channel': 'ip_min'})
    df = df.merge(gp, on=['ip','day','hour', 'minute'], how='left')
    df['ip_min'] = df['ip_min'].astype('uint32')
    print( "ip_min max val = ", df.ip_min.max() )
    del gp
    gc.collect()
    
    print("Group by ip, day, hour, minute, app...")
    gp = df[['ip', 'day', 'hour', 'minute', 'app', 'channel']].groupby(by=['ip', 'day',
             'hour', 'minute', 'app'])[['channel']].count().reset_index().rename(index=str, 
             columns={'channel': 'ip_min_app'})
    df = df.merge(gp, on=['ip','day','hour','minute','app'], how='left')
    df['ip_min_app'] = df['ip_min_app'].astype('uint32')
    print( "ip_min_app max val = ", df.ip_min_app.max() )
    del gp
    gc.collect()
    
    print("Group by ip, day, hour, minute, os...")
    gp = df[['ip', 'day', 'hour', 'minute', 'os', 'channel']].groupby(by=['ip', 'day',
             'hour', 'minute', 'os'])[['channel']].count().reset_index().rename(index=str, 
             columns={'channel': 'ip_min_os'})
    df = df.merge(gp, on=['ip','day','hour','minute','os'], how='left')
    df['ip_min_os'] = df['ip_min_os'].astype('uint32')
    print( "ip_min_os max val = ", df.ip_min_os.max() )
    del gp
    gc.collect()
    
    # print("Group by ip, day, hour, minute, device...")
    # gp = df[['ip', 'day', 'hour', 'minute', 'device', 'channel']].groupby(by=['ip', 'day',
    #          'hour', 'minute', 'device'])[['channel']].count().reset_index().rename(index=str, 
    #          columns={'channel': 'ip_min_dev'})
    # df = df.merge(gp, on=['ip','day','hour','minute','device'], how='left')
    # df['ip_min_dev'] = df['ip_min_dev'].astype('uint32')
    # print( "ip_min_dev max val = ", df.ip_min_dev.max() )
    # del gp
    # gc.collect()
    
    # print("Aggregation of features inside second...")
    
    # print("Group by ip, day, hour, minute, second...")
    # gp = df[['ip', 'day', 'hour', 'minute', 'second', 'channel']].groupby(by=['ip', 'day',
    #          'hour','minute', 'second'])[['channel']].count().reset_index().rename(index=str, 
    #          columns={'channel': 'ip_sec'})
    # df = df.merge(gp, on=['ip','day','hour', 'minute', 'second'], how='left')
    # df['ip_sec'] = df['ip_sec'].astype('uint32')
    # print( "ip_sec max val = ", df.ip_sec.max() )
    # del gp
    # gc.collect()

    print("Other agregations...")

    print("Group by device, app...")
    gp = df[['device','app','channel']].groupby(by=['device',
            'app'])[['channel']].count().reset_index().rename(index=str, 
             columns={'channel': 'device_app'})
    df = df.merge(gp, on=['device','app'], how='left')
    df['device_app'] = df['device_app'].astype('uint32')
    print( "device_app max val = ", df.device_app.max() )
    del gp
    gc.collect()

    
    print( df.info(memory_usage='deep') )

    df.drop( ['ip','day', 'minute', 'second'], axis=1, inplace=True )
    gc.collect()
    print( df.info(memory_usage='deep') )
    
    return( df )

#---------------------------------------------------------------------------------

print('Load train set...')
print('%.2f seconds' % (time.time()-start))
train_df = pd.read_csv(path+"train.csv", dtype=dtypes, usecols=train_cols)

train_df = filtrationByDateTrain(train_df)
gc.collect()

print( "Train info before: ")
print( train_df.info(memory_usage='deep') )
print('%.2f seconds' % (time.time()-start))
train_df = prep_data( train_df )
gc.collect()

print( "Train info after: ")
print( train_df.info(memory_usage='deep') )

print("vars and data type: ")
train_df.info(memory_usage='deep')

metrics = 'auc'
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric':metrics,
    'learning_rate': 0.1,
    'num_leaves': 9,  # we should let it be smaller than 2^(max_depth)
    'max_depth': 5,  # -1 means no limit
    'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 100,  # Number of bucketed bin for feature values
    'subsample': 0.9,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
    'nthread': 8,
    'verbose': 0,
    'scale_pos_weight':300, # because training data is extremely unbalanced 
    'metric':metrics
}

print(train_df.head(5))

print('Load valid set...')
print('%.2f seconds' % (time.time()-start))
val_df = pd.read_csv(path+"train.csv", dtype=dtypes, usecols=train_cols)
val_df = filtrationByDateValid(val_df)
print('%.2f seconds' % (time.time()-start))
val_df = prep_data( val_df )
gc.collect()

print(train_df.info(memory_usage='deep'))
print(val_df.info(memory_usage='deep'))

print("train size: ", len(train_df))
print("valid size: ", len(val_df))

gc.collect()

print("Training...")
print('%.2f seconds' % (time.time()-start))
num_boost_round=MAX_ROUNDS
early_stopping_rounds=EARLY_STOP

xgtrain = lgb.Dataset(train_df[predictors].values, label=train_df[target].values,
                        feature_name=predictors,
                        categorical_feature=categorical
                        )
del train_df
gc.collect()

xgvalid = lgb.Dataset(val_df[predictors].values, label=val_df[target].values,
                        feature_name=predictors,
                        categorical_feature=categorical
                        )
del val_df
gc.collect()

evals_results = {}

bst = lgb.train(lgb_params, 
                    xgtrain, 
                    valid_sets= [xgvalid], 
                    valid_names=['valid'], 
                    evals_result=evals_results, 
                    num_boost_round=num_boost_round,
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=10, 
                    feval=None)

n_estimators = bst.best_iteration

print("\nModel Report")
print("n_estimators : ", n_estimators)
print(metrics+":", evals_results['valid'][metrics][n_estimators-1])

del xgvalid
del xgtrain
gc.collect()

print('Load test set...')
print('%.2f seconds' % (time.time()-start))
test_df = pd.read_csv(path+"test.csv", dtype=dtypes, usecols=test_cols)
print("Converting to datetime...")
test_df['click_time'] = pd.to_datetime(test_df['click_time'])
test_df = prep_data( test_df )
gc.collect()

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id']

print("Predicting...")
sub['is_attributed'] = bst.predict(test_df[predictors])
print("Feature importance...")
print(bst.feature_name())
print(bst.feature_importance())
print("writing...")
sub.to_csv(outfile, index=False, float_format='%.9f')
print("done...")
print('%.2f seconds' % (time.time()-start))
print(sub.info(memory_usage='deep'))