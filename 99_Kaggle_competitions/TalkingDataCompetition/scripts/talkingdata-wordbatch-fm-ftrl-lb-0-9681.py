import sys

#sys.path.insert(0, '../input/wordbatch/')
sys.path.insert(0, '../input/randomstate/')
import wordbatch
from wordbatch.extractors import WordHash
from wordbatch.models import FM_FTRL
import threading
import pandas as pd
from sklearn.metrics import roc_auc_score
import time
import numpy as np

from contextlib import contextmanager
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print("[{}] done in {} s".format(name, t0))

start_time = time.time()

mean_auc= 0

def fit_batch(clf, X, y, w):  clf.partial_fit(X, y, sample_weight=w)

def predict_batch(clf, X):  return clf.predict(X)

def evaluate_batch(clf, X, y, rcount):
    auc= roc_auc_score(y, predict_batch(clf, X))
    global mean_auc
    if mean_auc==0:
        mean_auc= auc
    else: mean_auc= 0.2*(mean_auc*4 + auc)
    print(rcount, "ROC AUC:", auc, "Running Mean:", mean_auc)
    return auc

def df_add_counts(df, cols):
    arr_slice = df[cols].values
    unq, unqtags, counts = np.unique(np.ravel_multi_index(arr_slice.T, arr_slice.max(0) + 1),
                                     return_inverse=True, return_counts=True)
    df["_".join(cols)+'_count'] = counts[unqtags]

def df2csr(wb, df, pick_hours=None):
    df['datetime'] = pd.to_datetime(df.click_time)
    df.reset_index(drop=True, inplace=True)
    df= pd.concat([df, pd.DataFrame(df['datetime'].apply(lambda x: #str(x).split(" ")[0].split("-")).tolist(),
                                     str(x).replace(" ", ":").replace("-", ":").split(":")).tolist(),
                                   columns = ["year", "month", "dom", "hour", "min", "sec"])], axis= 1)
    df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
    if 'is_attributed' in df.columns:
        labels = df['is_attributed'].values
        weights = np.multiply([1.0 if x == 1 else 0.2 for x in df['is_attributed'].values],
                          df['hour'].apply(lambda x: 1.0 if x in pick_hours else 0.5))
    else:
        labels= []
        weights= []
    df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')
    df.drop(['sec', 'year', 'month', 'min', 'click_time', 'datetime'], axis=1, inplace=True)

    df_add_counts(df, ['ip', 'day', 'hour'])
    df_add_counts(df, ['ip', 'app'])
    df_add_counts(df, ['ip', 'app', 'os'])
    df_add_counts(df, ['ip', 'device'])
    df_add_counts(df, ['app', 'channel'])
    df['ip_day_hour_count']= np.log2(1+df['ip_day_hour_count'].values).astype(int)
    df['ip_app_count']= np.log2(1+df['ip_app_count'].values).astype(int)
    df['ip_app_os_count']= np.log2(1+df['ip_app_os_count'].values).astype(int)
    df['ip_device_count']= np.log2(1+df['ip_device_count'].values).astype(int)
    df['app_channel_count']= np.log2(1+df['app_channel_count'].values).astype(int)
    tmp= "XI" + df['ip'].astype(str) \
        + " XA" + df['app'].astype(str) \
        + " XD" + df['device'].astype(str) \
        + " XO" + df['os'].astype(str) \
        + " XC" + df['channel'].astype(str) \
        + " XWD" + df['day'].astype('str') \
        + " XH" + df['hour'].astype('str') \
        + " XAXC" + df['app'].astype('str')+"_"+df['channel'].astype('str') \
        + " XOXC" + df['os'].astype('str')+"_"+df['channel'].astype('str') \
        + " XAXD" + df['app'].astype('str')+"_"+df['device'].astype('str') \
        + " XAXOCXC" + df['app'].astype('str')+"_"+df['os'].astype('str') \
          +"_"+df['channel'].astype('str') \
        + " XIXA" + df['ip'].astype('str')+"_"+df['app'].astype('str') \
        + " XAXD" + df['app'].astype('str')+"_"+df['device'].astype('str') \
        + " XAXO" + df['app'].astype('str')+"_"+df['os'].astype('str') \
        + " XIDHC" + df['ip_day_hour_count'].astype('str') \
        + " XIAC" + df['ip_app_count'].astype('str') \
        + " XAOC" + df['ip_app_os_count'].astype('str') \
        + " XIDC" + df['ip_device_count'].astype('str') \
        + " XAC" + df['app_channel_count'].astype('str')
    del(df)
    return wb.transform(tmp), labels, weights

class ThreadWithReturnValue(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
        threading.Thread.__init__(self, group, target, name, args, kwargs, daemon=daemon)
        self._return = None
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)
    def join(self):
        threading.Thread.join(self)
        return self._return

batchsize = 10000000
D = 2 ** 22

wb = wordbatch.WordBatch(None, extractor=(WordHash, {"ngram_range": (1, 1), "analyzer": "word",
                                                     "lowercase": False, "n_features": D,
                                                     "norm": None, "binary": True})
                         , minibatch_size=batchsize // 80, procs=8, verbose=0)
clf = FM_FTRL(alpha=0.05, beta=0.1, L1=0.0, L2=0.0, D=D, alpha_fm=0.02,
              L2_fm=0.0, init_fm=0.01, weight_fm=1.0,
              D_fm=8, e_noise=0.0, iters=3,
              inv_link="sigmoid", threads=4
              )
p = None
rcount = 0
for df_c in pd.read_csv('../input/train.csv', engine='c', chunksize=batchsize,
#for df_c in pd.read_csv('../input/train.csv', engine='c', chunksize=batchsize,
                        sep=","):
    rcount += batchsize
    X, labels, weights= df2csr(wb, df_c, pick_hours={4, 5, 10, 13, 14})
    if X.shape[1]==0: continue
    if rcount % (2 * batchsize) == 0:
        if p != None:  p.join()
        p = threading.Thread(target=evaluate_batch, args=(clf, X, labels, rcount))
        p.start()
    print("Training", rcount, time.time() - start_time)
    if p != None:  p.join()
    p = threading.Thread(target=fit_batch, args=(clf, X, labels, weights))
    p.start()
if p != None:  p.join()

p = None
click_ids= []
test_preds = []
rcount = 0
for df_c in pd.read_csv('../input/test.csv', engine='c', chunksize=batchsize,
#for df_c in pd.read_csv('../input/test.csv', engine='c', chunksize=batchsize,
                        sep=","):
    rcount += batchsize
    X, labels, weights = df2csr(wb, df_c)
    if rcount % (10 * batchsize) == 0:
        print(rcount)
    if p != None:  test_preds += list(p.join())
    p = ThreadWithReturnValue(target=predict_batch, args=(clf, X))
    p.start()
    click_ids+= df_c['click_id'].tolist()
if p != None:  test_preds += list(p.join())

df_sub = pd.DataFrame({"click_id": click_ids, 'is_attributed': test_preds})
df_sub.to_csv("../output/wordbatch_fm_ftrl.csv", index=False)
