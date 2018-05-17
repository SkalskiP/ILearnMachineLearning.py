from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import gc
from .utils import show_features, do_next_click, do_prev_click, do_var, do_count, do_countuniq, do_cumcount, add_counts, \
    do_countuniq_and_save, do_count_and_save, do_mean, do_next_click_group, do_prev_click_group, show_gain
from .time_tracker import TimeTracker

# ======================================================================================================================
# Model settings
# ======================================================================================================================

os.environ['OMP_NUM_THREADS'] = '1'

model_v = 0

# RUN SCRIPT
max_rounds = 200
early_stop = 50

train_file = "train50M.h5"
train_object_name = 'train_df_50'

# RUN DEBUG
# max_rounds = 10
# early_stop = 10
#
# train_file = "train10M.h5"
# train_object_name = 'train_df_10'

path = "../../input/"

feature_importance_file = "features_importance_v4_{}.txt"
feature_gain_file = "features_gain_v4.txt"

dtypes = {
    'ip':               'uint32',
    'app':              'uint16',
    'device':           'uint16',
    'os':               'uint16',
    'channel':          'uint16',
    'is_attributed':    'uint8',
    'click_id':         'uint32',
}

train_cols =        ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_cols =         ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']
common_cols =       ['ip', 'app', 'device', 'os', 'channel', 'click_time']
categorical_cols =  ['app', 'device', 'os', 'channel', 'hour']
excluded_features = ['ip']
target = 'is_attributed'

# MODEL SETTINGS

metrics = 'auc'
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': metrics,
    'learning_rate': .2,
    'num_leaves': 9,
    'max_depth': 4,
    'min_child_samples': 100,
    'max_bin': 200,
    'subsample': 0.7,
    'subsample_freq': 1,
    'colsample_bytree': 0.9,
    'min_child_weight': 0,
    'min_split_gain': 0,
    'nthread': 4,
    'verbose': 1,
    'scale_pos_weight': 300,
    'subsample_for_bin': 200000,
}

most_freq_hours_in_test_data = [4, 5, 9, 10, 13, 14]
least_freq_hours_in_test_data = [6, 11, 15]

# MODELS TO TEST

features_to_test = [

    # V1
    "do_next_click(train_df, ['ip', 'os', 'device', 'app'])",
    "do_next_click(train_df, ['ip', 'device'])",
    "do_next_click(train_df, ['ip', 'os', 'app'])",
    "do_next_click(train_df, ['ip', 'os', 'device'])",
    "do_next_click(train_df, ['ip', 'os', 'channel'])",
    "do_next_click(train_df, ['ip', 'channel'])",
    "do_next_click(train_df, ['ip', 'app', 'channel'])",
    "do_next_click(train_df, ['ip', 'app'])",

    # V2
    "do_prev_click(train_df, ['ip', 'device', 'app'])",
    "do_prev_click(train_df, ['ip', 'os', 'device', 'app'])",
    "do_prev_click(train_df, ['ip', 'app', 'channel'])",

    # V3
    "add_counts(train_df, ['ip'])",
    "add_counts(train_df, ['os', 'device'])",
    "add_counts(train_df, ['os', 'app', 'channel'])",
    "add_counts(train_df, ['app', 'channel'])",
    "add_counts(train_df, ['ip', 'app', 'os'])",
    "add_counts(train_df, ['in_test_hh', 'ip'])",

    # V4
    "do_countuniq(train_df, ['ip'], 'device')",
    "do_countuniq(train_df, ['ip'], 'app')",
    "do_countuniq(train_df, ['ip', 'device', 'os'], 'channel')",
    "do_countuniq(train_df, ['ip'], 'channel' )",

    # PREVIOUSLY TESTED

    # V1
    "do_next_click(train_df, ['ip', 'app', 'device', 'os', 'channel'])",
    "do_next_click(train_df, ['ip', 'device', 'channel'])",
    "do_next_click(train_df, ['ip', 'device', 'app'])",
    "do_next_click(train_df, ['ip', 'os'])",

    # V2
    "do_prev_click(train_df, ['ip', 'app', 'device', 'os', 'channel'])",
    "do_prev_click(train_df, ['ip', 'os', 'device'])",
    "do_prev_click(train_df, ['ip', 'os', 'app'])",
    "do_prev_click(train_df, ['ip', 'os', 'channel'])",
    "do_prev_click(train_df, ['ip', 'device', 'channel'])",
    "do_prev_click(train_df, ['ip', 'os'])",
    "do_prev_click(train_df, ['ip', 'channel'])",
    "do_prev_click(train_df, ['ip', 'app'])",
    "do_prev_click(train_df, ['ip', 'device'])",

    # V3
    "add_counts(train_df, ['ip'])",
    "add_counts(train_df, ['ip', 'device'])",
    "add_counts(train_df, ['app'])",
    "add_counts(train_df, ['ip', 'day', 'in_test_hh'])",
    "add_counts(train_df, ['ip', 'day', 'hour'])",
    "add_counts(train_df, ['ip', 'os', 'day', 'hour'])",
    "add_counts(train_df, ['ip', 'app', 'day', 'hour'])",
    "add_counts(train_df, ['ip', 'device', 'day', 'hour'])",
    "add_counts(train_df, ['day', 'hour', 'app'])",

    # V4
    "do_countuniq(train_df, ['ip'], 'os')",
    "do_countuniq(train_df, ['app'], 'channel')",
    "do_countuniq(train_df, ['app'], 'os')",
    "do_countuniq(train_df, ['app'], 'ip')",
    "do_countuniq(train_df, ['app'], 'device')",
    "do_countuniq(train_df, ['ip', 'day'], 'channel' )",
    "do_countuniq(train_df, ['ip', 'day'], 'app' )",
    "do_countuniq(train_df, ['ip', 'day'], 'device' )",
    "do_countuniq(train_df, ['ip', 'day'], 'os' )",
    "do_countuniq(train_df, ['ip', 'app'], 'os')",
    "do_countuniq(train_df, ['ip', 'app'], 'device')",
    "do_countuniq(train_df, ['ip', 'app'], 'channel')",
    "do_countuniq(train_df, ['ip', 'device', 'os'], 'app')",

    # V5
    "do_count(train_df, ['ip', 'app'])",
    "do_count(train_df, ['ip', 'os'])",
    "do_count(train_df, ['ip', 'device'])",
    "do_count(train_df, ['ip', 'channel'])",
    "do_count(train_df, ['app', 'channel'])",
    "do_count(train_df, ['app', 'os'])",
    "do_count(train_df, ['app', 'device'])",
    "do_count(train_df, ['ip', 'app', 'os'])",
    "do_count(train_df, ['ip', 'app', 'channel'])",
    "do_count(train_df, ['ip', 'app', 'device'])",
    "do_count(train_df, ['app', 'channel', 'os'])",
    "do_count(train_df, ['app', 'channel', 'device'])",

    # V6
    "do_var(train_df, ['ip'], 'os')",
    "do_var(train_df, ['app'], 'channel')",
    "do_var(train_df, ['app'], 'os')",
    "do_var(train_df, ['app'], 'ip')",
    "do_var(train_df, ['app'], 'device')",
    "do_var(train_df, ['ip', 'day'], 'channel' )",
    "do_var(train_df, ['ip', 'day'], 'app' )",
    "do_var(train_df, ['ip', 'day'], 'device' )",
    "do_var(train_df, ['ip', 'day'], 'os' )",
    "do_var(train_df, ['ip', 'app'], 'os')",
    "do_var(train_df, ['ip', 'app'], 'device')",
    "do_var(train_df, ['ip', 'app'], 'channel')",
    "do_var(train_df, ['ip', 'device', 'os'], 'app')",
    "do_var(train_df, ['ip'], 'device')",
    "do_var(train_df, ['ip'], 'app')",
    "do_var(train_df, ['ip', 'device', 'os'], 'channel')",
    "do_var(train_df, ['ip'], 'channel' )",
]

tested_features_names = []
tested_features_gains = []

# ======================================================================================================================
# Supporting methods definition and feature selection
# ======================================================================================================================


def initial_preproc(df):
    print('>> Initial preprocessing...')

    df['click_time'] = pd.to_datetime(df['click_time'])
    gc.collect()

    df['hour'] = df['click_time'].dt.hour.astype('uint8')

    df['day'] = df['click_time'].dt.day.astype('uint8')
    # not used as feature
    excluded_features.append('day')

    df['in_test_hh'] = (3
                        - 2 * df['hour'].isin(most_freq_hours_in_test_data)
                        - 1 * df['hour'].isin(least_freq_hours_in_test_data)).astype('uint8')
    # not used as feature
    excluded_features.append('in_test_hh')

    df['click_time'] = (df['click_time'].astype(np.int64) // 10 ** 9).astype(np.int32)
    # not used as feature
    excluded_features.append('click_time')
    gc.collect()

    return df


def train_model(X, Y, inputs, feature_importance_file):

    print('>> Train for {}...'.format(inputs))

    X_train, X_val = train_test_split(X, train_size=.9, shuffle=False)
    Y_train, Y_val = train_test_split(Y, train_size=.9, shuffle=False)

    print('>> Train size:', len(X_train))
    print('>> Valid size:', len(X_val))

    gc.collect()

    xgtrain = lgb.Dataset(X_train[inputs].values.astype(np.float32),
                          label=Y_train,
                          feature_name=inputs,
                          categorical_feature=categorical_cols)
    del X_train
    del Y_train
    gc.collect()

    xgvalid = lgb.Dataset(X_val[inputs].values.astype(np.float32),
                          label=Y_val,
                          feature_name=inputs,
                          categorical_feature=categorical_cols)
    del X_val
    del Y_val
    gc.collect()

    evals_results = {}

    model = lgb.train(lgb_params,
                      xgtrain,
                      valid_sets=[xgvalid],
                      valid_names=['valid'],
                      evals_result=evals_results,
                      num_boost_round=max_rounds,
                      early_stopping_rounds=early_stop,
                      verbose_eval=1,
                      feval=None)

    n_estimators = model.best_iteration

    print('\n>> Model Info:')
    print('n_estimators:', n_estimators)
    print(metrics + ':', evals_results['valid'][metrics][n_estimators - 1])

    del xgvalid
    del xgtrain
    gc.collect()

    print(">> Feature importance...")

    show_features(model.feature_name(), model.feature_importance(), to_file=True, file_name=feature_importance_file)

    return evals_results['valid'][metrics][n_estimators - 1]

# ======================================================================================================================
# Supporting methods definition
# ======================================================================================================================


tt = TimeTracker()
tt.start()


print('>> Loading train.h5...')
train_df = pd.read_hdf(path + train_file, train_object_name)

train_len = len(train_df)

print("\n>> Train set length: " + str(train_len))
print(train_df.head(5))


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

print('>> Preprocessing...')

# saving is_attributed values for later
y = train_df.is_attributed.values

train_df = initial_preproc(train_df)
gc.collect()

print('>> Train...')

# build set of features to be used during train
used_features = list(set(train_df.columns) - {target} - set(excluded_features))

# creating output file name and incrementing model count
file_name = feature_importance_file.format(model_v)
model_v += 1

best_score = train_model(train_df, y, used_features, file_name)

for new_feature in features_to_test:
    train_df, feature_name = eval(new_feature)
    gc.collect()

    # build set of features to be used during train
    used_features = list(set(train_df.columns) - {target} - set(excluded_features))

    # creating output file name and incrementing model count
    file_name = feature_importance_file.format(model_v)
    model_v += 1

    new_score = train_model(train_df, y, used_features, file_name)
    gc.collect()

    tested_features_names.append(feature_name)
    tested_features_gains.append(new_score - best_score)

    if model_v % 10 == 0:
        show_gain(tested_features_names, tested_features_gains, to_file=False)

    if best_score < new_score:
        print('>> Score of model improved... {} is added'.format(feature_name))
        best_score = new_score
    else:
        print('>> Score of model got worse... {} has been removed'.format(feature_name))
        train_df.drop([feature_name], axis=1, inplace=True)
        gc.collect()

# saving features gains to file
show_gain(tested_features_names, tested_features_gains, to_file=True, file_name=feature_gain_file)










