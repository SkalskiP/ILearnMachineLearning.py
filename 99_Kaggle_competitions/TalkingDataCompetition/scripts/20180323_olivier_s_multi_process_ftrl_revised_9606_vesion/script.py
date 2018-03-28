"""
Based on supernova's code : https://www.kaggle.com/supernova117
https://www.kaggle.com/supernova117/ftrl-with-validation-and-auc
Modified by olivier : https://www.kaggle.com/ogrellier
Addition : Multi processing with concurrent updates of weights
"""

from datetime import datetime
from math import exp, log, sqrt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from numba import jit
from multiprocessing import Process, Value, Array, Lock, Pool, cpu_count
import gc
import sys, os, psutil
from sklearn.metrics import mean_squared_error
from collections import defaultdict
import functools
import re
import unidecode
from itertools import combinations

np.random.seed(4689571)


class FTRLProximal(object):
    """ Our main algorithm: Follow the regularized leader - proximal

        In short,
        this is an adaptive-learning-rate sparse logistic-regression with
        efficient reg_alpha-reg_lambda-regularization

        Reference:
        http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
    """

    def __init__(self, id_name="id", target_name="target",
                 alpha=1e-2, beta=1.0, reg_alpha=1e-5,
                 reg_lambda=1.0, dim_expo=24,
                 interaction=False, n_jobs=2):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda

        self.id = id_name
        self.target_name = target_name

        # Check n_jobs
        if n_jobs == -1:
            self.cpus = cpu_count()
        else:
            self.cpus = min(n_jobs, cpu_count())

        # feature related parameters
        self.D = 2 ** dim_expo
        self.interaction = interaction

        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        self.n = np.zeros(self.D)  # [0.] * self.D
        self.z = np.random.uniform(0.0, 1.0, self.D)
        self.w = {}
        # Filters are not in use yet
        self.filter = defaultdict(int)
        self.min_occ = 2

    def _logloss(self, p, y):
        p = max(min(p, 1. - 10e-15), 10e-15)
        return -log(p) if y == 1. else -log(1. - p)

    def multi_fit(self, data, target, epochs=1, eval_x=None, eval_y=None):

        hash_x = data.copy()
        # Remove id if in the columns
        if self.id in hash_x:
            hash_x.drop(self.id, axis=1, inplace=True)

        if eval_x is not None:
            # val_hash_x = self.multi_hash(eval_x)
            val_hash_x = eval_x.copy()
            # Remove id if in the columns
            if self.id in val_hash_x:
                val_hash_x.drop(self.id, axis=1, inplace=True)

        start = datetime.now()
        count = len(data)
        n_ = Array('d', self.n, lock=False)
        z_ = Array('d', self.z, lock=False)
        lock_ = Lock()
        for e_ in range(epochs):
            # Compute predictions
            loss_train = Value('d', 0, lock=False)
            full_data = np.hstack((target.values.reshape(-1, 1), hash_x.values))
            # Here we use Process directly since map does not support shared objects
            # z and n will be updated wildly, no lock has been implemented
            processes = [
                Process(target=self._train_samples,
                        args=(partial_data, z_, n_, lock_, loss_train))
                for partial_data in np.array_split(full_data, self.cpus)
            ]

            for p in processes:
                p.start()

            while processes:
                processes.pop().join()

            if eval_x is not None:
                # Compute validation losses
                p = Pool(self.cpus)
                # Shared memory is not supported by map and z_ and n_ should be read only here
                # So we create np.arrays form shared memory objects
                oof_v = p.map(functools.partial(self._predict_samples, z_=np.array(z_), n_=np.array(n_)),
                              np.array_split(val_hash_x.values, self.cpus))
                val_preds = np.hstack(oof_v)
                p.close()
                p.join()
                val_logloss = log_loss(eval_y, val_preds)
                val_auc = roc_auc_score(eval_y, val_preds)
                # Display current training and validation losses
                # t_logloss stands for current train_logloss, v for valid
                print('time_used:%s\tepoch: %-4drows:%d\tt_logloss:%.5f\tv_logloss:%.5f\tv_auc:%.6f'
                      % (datetime.now() - start, e_, count + 1, (loss_train.value / count), val_logloss, val_auc))
                del val_preds
                del oof_v
                gc.collect()
            else:
                print('time_used:%s\tepoch: %-4drows:%d\tt_logloss:%.5f'
                      % (datetime.now() - start, e_, count + 1, (loss_train.value / count)))

            # del loss_v
            # print(z_)
            gc.collect()

        self.n = np.array(n_)
        self.z = np.array(z_)

    def _train_samples(self, y_data, z_, n_, lock_, loss_):
        # retrieve target and data
        target_ = y_data[:, 0]
        data_ = y_data[:, 1:]
        # print(data_[:10])
        # loss_train = 0.
        idx = np.arange(len(data_))
        np.random.shuffle(idx)
        y = target_[idx]
        for count, x in enumerate(data_[idx]):
            # Train on current sample
            p = self._train_single(x, y[count], z_, n_, lock_)
            # Compute cumulated loss
            loss_.value += self._logloss(p, y[count])
            # print(loss_.value)
        return loss_.value

    def _train_single(self, x, y, z_, n_, lock_):
        # First get the indices
        indices = [i_ for i_ in self._indices(x)]

        # Compute prediction - does not need a lock
        p, w = self._predict_single(x, z_, n_, indices)

        # Update weights
        # gradient under log loss, remember all x are equal to 1 once hashed
        # 1 + y = 2 for is_attribute = 1 and 0 for is_attribute = 0 
        # so increase gradient for positives (if I understoof Scirpus' idea correctly)
        g = (p - y) * (1 + y)

        # update z and n - this part needs a lock
        # Full update must be done under the lock
        # with lock_:
        for i in indices:
            # Increase index occurences
            # self.filter[i] += 1
            # If enough occurences then update
            # if self.filter[i] > self.min_occ:
            sigma = (sqrt(n_[i] + g * g) - sqrt(n_[i])) / self.alpha
            z_[i] += g - sigma * w[i]
            n_[i] += g * g

        return p

    def _predict_single(self, x, z_, n_, indices=None):
        """
        Get probability estimation for input x
        The input is expected to be hashed
        outputs the probability and individual values
        """

        # compute probability
        w = {}

        # wTx is the inner product of w and x
        wTx = 0.
        # print(x)
        for i in indices:
            # sign = -1. if z[i] < 0 else 1.  # get sign of z[i]

            # build w on the fly using z and n, hence the name - lazy weights
            # we are doing this at prediction instead of update time is because
            # this allows us for not storing the complete w
            # if sign * z[i] <= reg_alpha:
            if abs(z_[i]) <= self.reg_alpha:  # or (self.filter[i] <= self.min_occ):
                # w[i] vanishes due to reg_alpha regularization
                w[i] = 0.
            else:
                # apply prediction time reg_alpha, reg_lambda regularization to z and get w
                w[i] = (np.sign(z_[i]) * self.reg_alpha - z_[i]) / \
                       ((self.beta + sqrt(n_[i])) / self.alpha + self.reg_lambda)

            wTx += w[i]

        # bounded sigmoid function, this is the probability estimation
        proba = 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

        return proba, w

    def predict_proba(self, df_dat):
        # Compute validation losses
        p = Pool(self.cpus)
        # Shared memory is not supported by map and z_ and n_ should be read only here
        # So we create np.arrays form shared memory objects
        oof_v = p.map(functools.partial(self._predict_samples,
                                        z_=self.z,
                                        n_=self.n),
                      np.array_split(df_dat.values, self.cpus))
        preds = np.hstack(oof_v)
        p.close()
        p.join()

        return preds

    def _predict_samples(self, np_dat, z_, n_):
        preds = np.zeros(len(np_dat))
        for i, row in enumerate(np_dat):
            indices = [i_ for i_ in self._indices(row)]
            preds[i], w_ = self._predict_single(row, z_, n_, indices)
        return preds

    def _indices(self, x):
        """ A helper generator that yields the indices in x
            The purpose of this generator is to make the following
            code a bit cleaner when doing feature interaction.
            x[0] is the ip
            x[1] is the app
            x[2] is the device
            x[3] is the os
            x[4] is the channel
            x[5] is the time
            x[6] is day of week
            x[7] is day of year
            x[8] is week of year
            x[9] is days to the end of month
            x[10] is days to end of year
        """
        # print(x)

        # first yield index of the bias term
        yield 0
        # yield ip
        yield abs(hash("ip_" + str(x[0]))) % self.D
        # yield app
        yield abs(hash("app_" + str(x[1]))) % self.D
        # yield device
        yield abs(hash("dev_" + str(x[2]))) % self.D
        # yield os
        yield abs(hash("os_" + str(x[3]))) % self.D
        # yield channel
        yield abs(hash("channel_" + str(x[4]))) % self.D
        # Now yield time
        time1 = x[5].split()
        date = time1[0]
        time = time1[1]
        # First yield date
        pref = ["year_", "month_", "dom_"]
        for i_t, tok in enumerate(date.split("-")):
            yield abs(hash(pref[i_t] + str(tok))) % self.D
        # Then yield time
        pref = ["hour_", "min_", "sec_"]
        for i_t, tok in enumerate(date.split(":")):
            yield abs(hash(pref[i_t] + str(tok))) % self.D
        # Yield dow
        yield abs(hash("dow_" + str(x[6]))) % self.D
        # Yield doy
        yield abs(hash("doy_" + str(x[7]))) % self.D
        # Yield woy
        yield abs(hash("woy_" + str(x[8]))) % self.D
        # Yield remaining days in month
        yield abs(hash("dteom_" + str(x[9]))) % self.D
        # Yield remaining days in year
        yield abs(hash("dteoy_" + str(x[10]))) % self.D

        # Now yield combinations
        # App + Channel
        yield abs(hash("app_chan_" + str(x[1]) + "_" + str(x[4]))) % self.D
        # OS + Channel
        yield abs(hash("os_chan_" + str(x[3]) + "_" + str(x[4]))) % self.D
        # OS + Channel
        yield abs(hash("app_os_chan_" + str(x[1]) + str(x[3]) + "_" + str(x[4]))) % self.D


def cpuStats(disp=""):
    """ @author: RDizzl3 @address: https://www.kaggle.com/rdizzl3"""
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30
    print("%s MEMORY USAGE for PID %10d : %.3f" % (disp, pid, memoryUse))


def add_time_related_info(df):
    # Hour, minutes and seconds are already used by pure hashing
    # year, month and day of month as well
    df["datetime"] = pd.to_datetime(df["click_time"])
    df["dow"] = df["datetime"].dt.dayofweek
    df["doy"] = df["datetime"].dt.dayofyear
    df["woy"] = df["datetime"].dt.week
    df["dteom"] = df["datetime"].dt.daysinmonth - df["datetime"].dt.day
    daysinyear = df["doy"].mod(4).apply(lambda x: 365 if x>0 else 366)
    df["dteoy"] = daysinyear - df["doy"]
    # df["dteoy"] =
    del df["datetime"]
    del daysinyear
    gc.collect()

def main():
    train_file, trn_nb_samples, chunksize = "../../input/train.csv", 184903890, 20000000
    # train_file, trn_nb_samples, chunksize = "../input/train_sample.csv", 100000, 50000
    
    test_file = "../../input/test.csv"
    
    nb_epochs = 1
    
    dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8'
    }
    
    for epoch in range(nb_epochs):
        # Create FTRL model
        model = FTRLProximal(
            id_name="id",
            target_name="is_attributed",
            alpha=.05,  # Learning rate
            beta=0.1,  # Smoothing parameter for adaptive learning
            reg_alpha=0.,  # L1 regularization
            reg_lambda=0.,  # L2 regularization
            dim_expo=24,  # Hashing space dimension
            interaction=False,
            n_jobs=4
        )
        
        # Read train data in chuncks and train FTRL
        
        for i_c, df in enumerate(pd.read_csv(train_file, chunksize=chunksize, iterator=True, dtype=dtypes)):
            print("%10d / %10d read so far meaning %6.3f %%" 
                  % (i_c*chunksize, trn_nb_samples, 100*i_c*chunksize/trn_nb_samples))
            # Get target and drop it
            y = df["is_attributed"]
            del df["is_attributed"]
            
            add_time_related_info(df)
            
            features = [f_ for f_ in df 
                        if f_ not in ["click_id", "is_attributed", "attributed_time"]]
            
            # Train 
            model.multi_fit(
                df[features], y,
                epochs=3,
            )
            
    # Now read the test data and predict
    print("Reading test dataset")
    # sub = pd.read_csv(test_file, dtype=dtypes)
    # print(sub.shape)
    print("Predicting for test data")
    sub_preds = None
    sub_ids = None
    sub_nb_samples = 18790469
    chunksize = 5000000
    for i_c, df in enumerate(pd.read_csv(test_file, chunksize=chunksize, iterator=True, dtype=dtypes)):
        print("%10d / %10d read so far meaning %6.3f %%" 
              % (i_c*chunksize, sub_nb_samples, 100*i_c*chunksize/sub_nb_samples))
        add_time_related_info(df)
        if sub_preds is None:
            sub_preds = model.predict_proba(df[features])  # .values)
            sub_ids = df["click_id"].values
        else:
            sub_preds = np.hstack((sub_preds, model.predict_proba(df[features])))  # .values)))
            sub_ids = np.hstack((sub_ids, df["click_id"].values))
    print(sub_preds.shape)
    
    sub = pd.DataFrame()
    sub["is_attributed"] = sub_preds
    sub["click_id"] = sub_ids
    sub[["click_id", "is_attributed"]].to_csv("../../output/ftrl_submission.csv", index=False, float_format="%.6f")

if __name__ == '__main__':
    gc.enable()
    main()