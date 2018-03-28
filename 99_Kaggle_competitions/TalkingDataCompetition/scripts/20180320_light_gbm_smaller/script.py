# From Alexey Pronin's kernel:
#   https://www.kaggle.com/graf10a/lightgbm-lb-0-9675
# (now evised by Andy Harless attempting to squeeze out even more memory efficiency)

# THANK YOU AND ACKNOLEDGEMENTS:
# This kernel (Alexey's version) develops further the ideas suggested in:
#   *  "lgbm starter - early stopping 0.9539" by Aloisio Dourado, https://www.kaggle.com/aloisiodn/lgbm-starter-early-stopping-0-9539/code
#   * "LightGBM (Fixing unbalanced data)" by Pranav Pandya, https://www.kaggle.com/pranav84/lightgbm-fixing-unbalanced-data-auc-0-9787?scriptVersionId=2777211
#   * "LightGBM with count features" by Ravi Teja Gutta, https://www.kaggle.com/rteja1113/lightgbm-with-count-features
# I would like to extend my gratitude to these individuals for sharing their work.

# WHAT IS NEW IN ALEXEY's VERSION? 
# In addition to some cosmetic changes to the code/LightGBM parameters, I am adding the 'ip' feature to and 
# removing the 'day' feature from the training set, and using the last chunk of the training data to build the model.


VALIDATE = False

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc

path = '../../input/' 
path_train = path + 'train.csv'
path_test = path + 'test.csv'

train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }
        
skip = range(1, 80000000)

print('Loading the training data...')
train = pd.read_csv(path_train, skiprows=skip, dtype=dtypes, header=0, usecols=train_cols)
print('Loading the test data...')
test = pd.read_csv(path_test, dtype=dtypes, header=0, usecols=test_cols)

len_train = len(train)
print('The initial size of the train set is', len_train)
print('Binding the training and test set together...')
train=train.append(test)

target = 'is_attributed'
train.loc[train[target].isnull(),target] = 99
train[target] = train[target].astype('uint8')
train.info()


del test
gc.collect()

print("Creating new time features: 'hour' and 'day'...")
train['hour'] = pd.to_datetime(train.click_time).dt.hour.astype('uint8')
train['day'] = pd.to_datetime(train.click_time).dt.day.astype('uint8')

train.drop( 'click_time', axis=1, inplace=True )
gc.collect()

print("Creating new count features: 'n_channels', 'ip_app_count', 'ip_app_os_count'...")

print('Computing the number of channels associated with ')
print('a given IP address within each hour...')
n_chans = train[['ip','day','hour','channel']].groupby(by=['ip','day',
          'hour'])[['channel']].count().reset_index().rename(columns={'channel': 'n_channels'})
print('Merging the channels data with the main data set...')
train = train.merge(n_chans, on=['ip','day','hour'], how='left')
train['n_channels'] = train['n_channels'].astype('uint16')
del n_chans
gc.collect()

print('Computing the number of channels associated with ')
print('a given IP address and app...')
n_chans = train[['ip','app', 'channel']].groupby(by=['ip', 
          'app'])[['channel']].count().reset_index().rename(columns={'channel': 'ip_app_count'})
print('Merging the channels data with the main data set...')
train = train.merge(n_chans, on=['ip','app'], how='left')
train['ip_app_count'] = train['ip_app_count'].astype('uint16')
del n_chans
gc.collect()

print('Computing the number of channels associated with ')
print('a given IP address, app, and os...')
n_chans = train[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 
          'os'])[['channel']].count().reset_index().rename(columns={'channel': 'ip_app_os_count'})
print('Merging the channels data with the main data set...')       
train = train.merge(n_chans, on=['ip','app', 'os'], how='left')
train['ip_app_os_count'] = train['ip_app_os_count'].astype('uint16')
del n_chans
gc.collect()

train.info()

test = train[len_train:].copy().drop( target, axis=1 )
print('The size of the test set is ', len(test))



predictors = ['ip', 'device', 'app', 'os', 'channel', 'hour', 'n_channels', 'ip_app_count', 'ip_app_os_count']
categorical = ['ip', 'app', 'device', 'os', 'channel', 'hour']

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.1,
    'num_leaves': 255,  
    'max_depth': 8,  
    'min_child_samples': 100,  
    'max_bin': 100,  
    'subsample': 0.7,  
    'subsample_freq': 1,  
    'colsample_bytree': 0.7,  
    'min_child_weight': 0,  
    'subsample_for_bin': 200000,  
    'min_split_gain': 0,  
    'reg_alpha': 0,  
    'reg_lambda': 0,  
   # 'nthread': 8,
    'verbose': 0,
    'scale_pos_weight':99 
    }


if VALIDATE:
    
    r = 0.1 # the fraction of the train data to be used for validation
    val = train[(len_train-round(r*len_train)):len_train]
    print('The size of the validation set is ', len(val))

    train = train[:(len_train-round(r*len_train))]
    print('The size of the train set is ', len(train))


    gc.collect()

    print("Preparing the datasets for training...")

    
    dtrain = lgb.Dataset(train[predictors].values, label=train[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )
    del train
    gc.collect()

    dvalid = lgb.Dataset(val[predictors].values, label=val[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )
    del val
    gc.collect()
                      

    evals_results = {}

    print("Training the model...")

    lgb_model = lgb.train(params, 
                     dtrain, 
                     valid_sets=[dtrain, dvalid], 
                     valid_names=['train','valid'], 
                     evals_result=evals_results, 
                     num_boost_round=1000,
                     early_stopping_rounds=50,
                     verbose_eval=True, 
                     feval=None)
                     
    pred = lgb_model.predict(test[predictors], num_iteration=lgb_model.best_iteration)
                     
else:

    train = train[:len_train]
    print('The size of the train set is ', len(train))

    gc.collect()

    print("Preparing the datasets for training...")

    
    dtrain = lgb.Dataset(train[predictors].values, label=train[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )
    del train
    gc.collect()

    evals_results = {}

    print("Training the model...")

    num_iteration=390
    lgb_model = lgb.train(params, 
                     dtrain, 
                     valid_sets=[dtrain], 
                     valid_names=['train'], 
                     evals_result=evals_results, 
                     num_boost_round=num_iteration,
                     verbose_eval=True, 
                     feval=None)
                     
    pred = lgb_model.predict(test[predictors])
                     
# Feature names:
print('Feature names:', lgb_model.feature_name())

# Feature importances:
print('Feature importances:', list(lgb_model.feature_importance()))

print("Preparing data for submission...")

submit = pd.read_csv(path_test, dtype='int', usecols=['click_id'])

print("Predicting the submission data...")

submit['is_attributed'] = pred

print("Writing the submission data into a csv file...")

submit.to_csv('../../output/submission.csv', index=False)

print("All done...")