VALIDATE = False

MAX_ROUNDS = 1000
EARLY_STOP = 50
OPT_ROUNDS = 560

FULL_OUTFILE = 'sub_lgbm_r_to_python_nocv.csv'
VALID_OUTFILE = '../../output/sub_lgbm_r_to_python_withcv.csv'

import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split 
import lightgbm as lgb

path = '../../input/'

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

print('load train...')
train_cols = ['ip','app','device','os', 'channel', 'click_time', 'is_attributed']
train_df = pd.read_csv(path+"train.csv", skiprows=range(1,179903891), nrows=5000000,dtype=dtypes, usecols=train_cols)
# train_df = pd.read_csv(path+"train.csv", skiprows=range(1,109903891), nrows=75000000,dtype=dtypes, usecols=train_cols)

import gc

len_train = len(train_df)

gc.collect()

print('data prep...')

most_freq_hours_in_test_data = [4, 5, 9, 10, 13, 14]
least_freq_hours_in_test_data = [6, 11, 15]


def prep_data( df ):
    
    print("Converting to datetime...")
    df['click_time'] = pd.to_datetime(df['click_time'])
    print("Extract day and hour from datetime...")
    df['hour'] = df['click_time'].dt.hour.astype('uint8')
    df['day'] = df['click_time'].dt.dayofyear.astype('uint8')
    df.drop(['click_time'], axis=1, inplace=True)
    gc.collect()
    
    print("Marking as rush hours...")
    df['rush_hour'] = ( 1 +
                        2 * df['hour'].isin(most_freq_hours_in_test_data) -
                        1 * df['hour'].isin(least_freq_hours_in_test_data)).astype('uint8')

    print("Group by ip, day, hour, app...")
    gp = df[['ip', 'day', 'hour', 'app', 'channel']].groupby(by=['ip', 'day',
             'hour', 'app'])[['channel']].count().reset_index().rename(index=str, 
             columns={'channel': 'ip_day_hour_app'})
    df = df.merge(gp, on=['ip','day','hour','app'], how='left')
    df['ip_day_hour_app'] = df['ip_day_hour_app'].astype('uint32')
    print( "ip_day_hour_app max val = ", df.ip_day_hour_app.max() )
    del gp
    gc.collect()
    
    print("Group by ip, day, hour, device...")
    gp = df[['ip', 'day', 'hour', 'device', 'channel']].groupby(by=['ip', 'day',
             'hour', 'device'])[['channel']].count().reset_index().rename(index=str, 
             columns={'channel': 'ip_day_hour_div'})
    df = df.merge(gp, on=['ip','day','hour','div'], how='left')
    df['ip_day_hour_div'] = df['ip_day_hour_div'].astype('uint32')
    print( "ip_day_hour_div max val = ", df.ip_day_hour_div.max() )
    del gp
    gc.collect()

    df.drop( ['ip','day', 'hour', 'device', 'os'], axis=1, inplace=True )
    gc.collect()
    print( df.info() )
    
    return( df )

#---------------------------------------------------------------------------------

print( "Train info before: ")
print( train_df.info() )
train_df = prep_data( train_df )
gc.collect()
print( "Train info after: ")
print( train_df.info() )

print("vars and data type: ")
train_df.info()

metrics = 'auc'
lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric':metrics,
        'learning_rate': 0.1,
        'num_leaves': 7,  # we should let it be smaller than 2^(max_depth)
        'max_depth': 4,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'nthread': 8,
        'verbose': 0,
        'scale_pos_weight':99.7, # because training data is extremely unbalanced 
        'metric':metrics
}

target = 'is_attributed'
predictors = ['app', 'channel', 'rush_hour', 'ip_day_hour_app', 'ip_day_hour_div']
categorical = ['app', 'channel', 'rush_hour']

print(train_df.head(5))

if VALIDATE:

    train_df, val_df = train_test_split( train_df, train_size=.95, shuffle=False )

    print(train_df.info())
    print(val_df.info())

    print("train size: ", len(train_df))
    print("valid size: ", len(val_df))

    gc.collect()

    print("Training...")

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
    
    outfile = VALID_OUTFILE
    
    del xgvalid

else:

    print(train_df.info())

    print("train size: ", len(train_df))

    gc.collect()

    print("Training...")

    num_boost_round=OPT_ROUNDS

    xgtrain = lgb.Dataset(train_df[predictors].values, label=train_df[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )
    del train_df
    gc.collect()

    bst = lgb.train(lgb_params, 
                     xgtrain, 
                     num_boost_round=num_boost_round,
                     verbose_eval=10, 
                     feval=None)
                     
    outfile = FULL_OUTFILE

del xgtrain
gc.collect()

print('load test...')
test_cols = ['ip','app','device','os', 'channel', 'click_time', 'click_id']
test_df = pd.read_csv(path+"test.csv", dtype=dtypes, usecols=test_cols)

test_df = prep_data( test_df )
gc.collect()

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id']

print("Predicting...")
sub['is_attributed'] = bst.predict(test_df[predictors])
print("writing...")
sub.to_csv(outfile, index=False, float_format='%.9f')
print("done...")
print(sub.info())