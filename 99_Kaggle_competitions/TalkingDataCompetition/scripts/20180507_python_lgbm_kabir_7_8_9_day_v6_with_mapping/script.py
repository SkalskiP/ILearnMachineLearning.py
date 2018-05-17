from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import gc
from .utils import show_features, do_next_click, do_prev_click, do_var, do_count, do_countuniq, do_cumcount, do_mean
from .time_tracker import TimeTracker

os.environ['OMP_NUM_THREADS'] = '4'

script_version = 6
max_rounds = 1000
early_stop = 50
opt_rounds = 680
tt = TimeTracker()
tt.start()

output_file = 'lgbm_submit_day7_8_with_mapping_{}.csv'.format(str(script_version))
#validation_output_file = "20180502_validation_day7_8_train_mapping.csv"
feature_importance_file = "features_importance_{}.txt".format(str(script_version))

train_file = "train78day.h5"
test_file = "test.h5"
validation_file = "valid.h5"
test_supplement_file = "test_supplement.h5"

path = "../../input/"

dtypes = {
    'ip' :'uint32',
    'app' :'uint16',
    'device': 'uint16',
    'os' :'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8',
    'click_id': 'uint32',
}

train_cols = ['ip', 'app', 'device', 'os', 'channel', 'is_attributed', 'click_time']
test_cols = ['ip', 'app', 'device', 'os', 'click_time', 'channel', 'click_id']
common_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
categorical_cols = ['app', 'device', 'os', 'channel', 'hour']
join_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
target = 'is_attributed'

metrics = 'auc'
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': metrics,
    'learning_rate': .09,
    'num_leaves': 16,
    'max_depth': 5,
    'min_child_samples': 100,
    'max_bin': 300,
    'subsample': 0.7,
    'subsample_freq': 1,
    'colsample_bytree': 0.9,
    'min_child_weight': 0,
    'min_split_gain': 0,
    'nthread': 4,
    'verbose': 1,
    'scale_pos_weight': 325,
    'subsample_for_bin': 200000,
}

most_freq_hours_in_test_data = [4, 5, 9, 10, 13, 14]
least_freq_hours_in_test_data = [6, 11, 15]

# ======================================================================================================================
# Supporting methods definition and feature selection
# ======================================================================================================================


def add_counts(df, cols):
    arr_slice = df[cols].values
    unq, unqtags, counts = np.unique(np.ravel_multi_index(arr_slice.T, arr_slice.max(axis=0) + 1),
                                     return_inverse=True, return_counts=True)
    df["_".join(cols) + "_count"] = counts[unqtags]


def add_next_click(df):
    df['click_time_int'] = (df['click_time'].astype(np.int64) // 10 ** 9).astype(np.int32)
    df = do_next_click(df, agg_suffix='next_click')
    gc.collect()
    df = do_prev_click(df, agg_suffix='prev_click')
    gc.collect()

    return df


def preproc_data(df):
    print('>> Extrace date info...')

    df['click_time'] = pd.to_datetime(df['click_time'])
    df['hour'] = df['click_time'].dt.hour.astype('uint8')  # VERY GOOD
    df['day'] = df['click_time'].dt.day.astype('uint8')
    gc.collect()

    print(">> Hour bins...")

    df['in_test_hh'] = (3
                        - 2 * df['hour'].isin(most_freq_hours_in_test_data)
                        - 1 * df['hour'].isin(least_freq_hours_in_test_data)).astype('uint8')

    print('>> Adding next_click...')
    print(tt.get_time_from_start())

    add_next_click(df)

    print('>> Grouping...')
    print(tt.get_time_from_start())

    add_counts(df, ['ip'])
    add_counts(df, ['os', 'device'])
    add_counts(df, ['app', 'channel'])
    add_counts(df, ['ip', 'app', 'os'])
    add_counts(df, ['in_test_hh', 'ip'])
    add_counts(df, ['os', 'app', 'channel'])
    add_counts(df, ['ip', 'day', 'hour'])
    add_counts(df, ['ip', 'device', 'day', 'hour'])

    df = do_count(df, ['ip', 'device'])  # OK
    gc.collect()
    df = do_count(df, ['ip', 'channel'])
    gc.collect()
    df = do_count(df, ['ip', 'app', 'device'])
    gc.collect()
    df = do_count(df, ['ip', 'app'])  # OK
    gc.collect()
    df = do_count(df, ['ip', 'os'])
    gc.collect()
    df = do_count(df, ['ip', 'app', 'channel'])
    gc.collect()
    df = do_countuniq(df, ['ip'], 'channel')  # OK
    gc.collect()
    df = do_countuniq(df, ['ip', 'day'], 'channel')  # OK
    gc.collect()
    df = do_countuniq(df, ['ip'], 'device')
    gc.collect()
    df = do_countuniq(df, ['ip'], 'app')
    gc.collect()
    df = do_countuniq(df, ['ip'], 'os')
    gc.collect()
    df = do_countuniq(df, ['ip', 'app'], 'os')
    gc.collect()
    df = do_countuniq(df, ['ip', 'device', 'os'], 'app')
    gc.collect()
    df = do_var(df, ['ip'], 'device')
    gc.collect()
    df = do_var(df, ['ip', 'device', 'os'], 'channel')
    gc.collect()

    # can't drop 'ip' and 'click_time', those features are needed later on for merge of test_supplement and test
    df.drop(['day', 'in_test_hh', 'click_time_int'], axis=1, inplace=True)
    gc.collect()

    print(df.info())

    return df

# ======================================================================================================================
# Script
# ======================================================================================================================

# LOADING DATA FROM HARD DRIVE


print('>> Loading data...')
print(tt.get_time_from_start())

print('>> Loading train78day.h5...')
train_df = pd.read_hdf(path + train_file, 'df_day_78', dtype=dtypes, usecols=train_cols)

print('>> Loading valid.h5...')
val_df = pd.read_hdf(path + validation_file, 'df', dtype=dtypes, usecols=train_cols)

print('>> Load test_supplement.h5...')
test_supplement_df = pd.read_hdf(path + test_supplement_file, 'test_supplement_df', dtype=dtypes, usecols=test_cols)

# PRINT INFORMATION ABOUT LOADED DATA SETS


train_len = len(train_df)
val_len = len(val_df)
test_supplement_len = len(test_supplement_df)

print("\n>> Train set length: " + str(train_len))
print(train_df.head(5))

print("\n>> Validation set length: " + str(val_len))
print(val_df.head(5))

print("\n>> Test supplement set length: " + str(test_supplement_len))
print(test_supplement_df.head(5))

# DATA PREPROCESSING


print('>> Preprocessing...')
print(tt.get_time_from_start())


# saving is_attributed values from test and validation sets for later
y_train = train_df.is_attributed.values
y_val = val_df.is_attributed.values

# building single data set to conduct preprocessing at once
full_df = pd.concat([train_df[common_cols], val_df[common_cols], test_supplement_df[common_cols]])
gc.collect()

# actual preprocessing
full_df = preproc_data(full_df)
gc.collect()

# splitting full data set back into three separate ones
train_df = full_df.iloc[:train_len]
val_df = full_df.iloc[train_len:train_len+val_len]
test_supplement_df = full_df.iloc[train_len + val_len:]
gc.collect()

# names of features that will be used during model training
print('>> Inputs columns...')
inputs = list(set(train_df.columns) - {target} - {'ip', 'click_time'})
print(inputs)

# TRAINING


print('>> Training...')
print(tt.get_time_from_start())

print('>> Train size:', len(train_df))
print('>> Valid size:', len(val_df))


num_boost_round = max_rounds
early_stopping_rounds = early_stop


xgtrain = lgb.Dataset(train_df[inputs].values.astype(np.float32), label=y_train,
                      feature_name=inputs,
                      categorical_feature=categorical_cols)
del train_df
gc.collect()


xgvalid = lgb.Dataset(val_df[inputs].values.astype(np.float32), label=y_val,
                      feature_name=inputs,
                      categorical_feature=categorical_cols)
del val_df
gc.collect()

evals_results = {}

model = lgb.train(lgb_params,
                  xgtrain,
                  valid_sets=[xgvalid],
                  valid_names=['valid'],
                  evals_result=evals_results,
                  num_boost_round=num_boost_round,
                  early_stopping_rounds=early_stopping_rounds,
                  verbose_eval=1,
                  feval=None)

n_estimators = model.best_iteration

print('\n>> Model Info:')
print('n_estimators:', n_estimators)
print(metrics + ':', evals_results['valid'][metrics][n_estimators - 1])

del xgvalid
del xgtrain
gc.collect()

# PREDICTIONS


print(">> Feature importance...")
show_features(model.feature_name(), model.feature_importance(), to_file=True, file_name=feature_importance_file)

print('>> Predicting the submission data...')
print(tt.get_time_from_start())

test_supplement_df['is_attributed'] = model.predict(test_supplement_df[inputs], num_iteration=model.best_iteration)

print('>> Projecting prediction onto test...')
all_cols = join_cols + ['is_attributed']

print('>> Load test.h5...')
test_df = pd.read_hdf(path + test_file, 'test_df',  dtype=dtypes, usecols=test_cols)
test_df['click_time'] = pd.to_datetime(test_df['click_time'])


print('>> Test info...')
print(test_df.info())
print(test_df.head())
print('>> Test supplement info...')
print(test_supplement_df.info())
print(test_supplement_df.head())

test_df = test_df.merge(test_supplement_df[all_cols], how='left', on=join_cols)

print('>> Test info after marge...')
print(test_df.info())
print(test_df.head())

test_df = test_df.drop_duplicates(subset=['click_id'])

print('>> Test info after drop dup...')
print(test_df.info())
print(test_df.head())

print(">> Writing the submission data into a csv file...")

test_df[['click_id', 'is_attributed']].to_csv(output_file, index=False, float_format='%.9f')

print('Done!')





