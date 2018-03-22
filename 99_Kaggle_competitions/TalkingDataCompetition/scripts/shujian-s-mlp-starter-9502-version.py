## Reference:
# https://www.kaggle.com/georsara1/95-auc-score-in-train-sample-with-neural-nets
# https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
# https://www.kaggle.com/shujian/s2-fm-ftrl-cnn-efficient-v8-final
# https://www.kaggle.com/danofer/downsampling-for-fun-speed


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
np.random.seed(697)
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Modified from: https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
# Also refered:
# https://www.kaggle.com/georsara1/95-auc-score-in-train-sample-with-neural-nets

import os; os.environ['OMP_NUM_THREADS'] = '4'
from contextlib import contextmanager
from functools import partial
from operator import itemgetter
from multiprocessing.pool import ThreadPool
import time
from typing import List, Dict

from scipy.sparse import csr_matrix, hstack
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras import optimizers
import dask.dataframe as dd
import gc

dtypes = {
        'ip':'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'uint8'
        }

# Self made one hot encoding, can save the dict
# https://www.kaggle.com/shujian/s2-fm-ftrl-cnn-efficient-v8-final
def fit_dummy(list_data):
    n_data = len(list_data)
    n_cat = 0
    d_data = {}
    for cat in list_data:
        if cat not in d_data:
            d_data[cat] = n_cat
            n_cat += 1
    d_data['<NOT_SHOWN>'] = n_cat
    res = np.zeros(shape=(n_data, n_cat + 1))
    for i, v in enumerate(list_data):
        res[i, d_data[v]] = 1
    return csr_matrix(res), d_data
    
def transform_dummy(list_data, d_data):
    n_data = len(list_data)
    n_cat = len(d_data)
    res = np.zeros(shape=(n_data, n_cat))
    for i, v in enumerate(list_data):
        if v in d_data:
            res[i, d_data[v]] = 1
        else:
            res[i, n_cat - 1] = 1
    return csr_matrix(res)



@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df['time_month'] = df.click_time.str[5:7]
    df['time_day']   = df.click_time.str[8:10]
    df['time_hr']    = df.click_time.str[11:13]
    df['time_min']   = df.click_time.str[14:16]
    df['time_sec']   = df.click_time.str[17:20]
    # return df[['ip', 'app', 'device', 'os', 'channel', 'time_month', 'time_day', 'time_hr', 'time_min', 'time_sec']]
    return df[['app', 'device', 'os', 'channel', 'time_hr']]

def main():
    with timer('process train'):
        # train = pd.read_csv('../input/train_sample.csv')
        # train = pd.read_csv('../input/train.csv', nrows = 1e6)
        
        dask_df = dd.read_csv('../input/train.csv',dtype=dtypes)
        df_pos = dask_df[(dask_df['is_attributed'] == 1)].compute()
        df_neg = dask_df[(dask_df['is_attributed'] == 0)].compute()
        df_pos = df_pos.sample(n=5000)
        df_neg = df_neg.sample(n=2000000)
        train = pd.concat([df_pos,df_neg]).sample(frac=1)
        print(len(train))
        del dask_df, df_pos, df_neg; gc.collect()
        
        y_train = train['is_attributed'].values.reshape(-1, 1) 
        train = preprocess(train)
        X_app, d_app = fit_dummy(train['app'].tolist())
        X_device, d_device = fit_dummy(train['device'].tolist())
        X_os, d_os = fit_dummy(train['os'].tolist())
        X_channel, d_channel = fit_dummy(train['channel'].tolist())
        X_time_hr, d_time_hr = fit_dummy(train['time_hr'].tolist())
        train_sparse = hstack((X_app, X_device, X_os, X_channel, X_time_hr)).tocsr()
        
        X_train, X_valid, y_train, y_valid = train_test_split(train_sparse, y_train, test_size = 0.2, random_state= 1984, stratify = y_train)

        print(f'X_train: {X_train.shape} of {X_train.dtype}')
        del train, train_sparse; gc.collect()

    with timer('train and val'):
        model = Sequential()
        model.add(Dense(48, input_dim=X_train.shape[1], kernel_initializer='normal', activation="tanh"))
        model.add(Dropout(0.5))
        model.add(Dense(24, activation="tanh"))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation("sigmoid"))
        model.compile(loss='binary_crossentropy', optimizer='adam')
        for i in range(5):
            with timer(f'epoch {i + 1}'):
                model.fit(x=X_train, y=y_train, batch_size=64, epochs=1, verbose=2)
     
        y_val_pred = model.predict(X_valid)[:, 0]
        print('Valid AUC: {:.4f}'.format(roc_auc_score(y_valid, y_val_pred)))
        del X_train, X_valid, y_train, y_valid; gc.collect()
    

    with timer('process test'):   
        y_preds = []
        for test in pd.read_csv('../input/test.csv', chunksize= 2e6):
            test = preprocess(test)
            X_app_test = transform_dummy(test['app'].tolist(), d_app)  
            X_device_test = transform_dummy(test['device'].tolist(), d_device)  
            X_os_test = transform_dummy(test['os'].tolist(), d_os)  
            X_channel_test = transform_dummy(test['channel'].tolist(), d_channel)  
            X_time_hr_test = transform_dummy(test['time_hr'].tolist(), d_time_hr)  
            
            X_test = hstack((X_app_test, X_device_test, X_os_test, X_channel_test, X_time_hr_test)).tocsr()
            y_preds += model.predict(X_test)[:, 0].tolist()

    with timer('test'):
        
        sub = pd.read_csv("../input/sample_submission.csv")
        sub['is_attributed'] = y_preds
        sub.to_csv('../output/sub_mlp.csv', index=False)
        print(sub.head())

if __name__ == '__main__':
    main()