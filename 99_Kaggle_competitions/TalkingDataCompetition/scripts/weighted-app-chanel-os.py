import pandas as pd

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8'
        }

test_df   = pd.read_csv('../input/test.csv', dtype=dtypes)
train_df  = pd.read_csv('../input/train.csv', dtype=dtypes)
print(train_df.shape, test_df.shape)

def predict(train, test, attrs=[]):
    return pd.merge(test, train.groupby(attrs)['is_attributed'].mean().reset_index(), on=attrs, how='left').fillna(0).set_index('click_id')

train_df.drop(['ip'], axis=1, inplace=True)
test_df.drop(['ip'], axis=1, inplace=True)

#https://www.kaggle.com/mapodoufu/baseline-the-channel-is-important-for-download
df_app_chnnl = predict(train_df, test_df, attrs=['app', 'channel'])
#https://www.kaggle.com/danofer/baseline-app-mean-94-1-auc-lb
df_app       = predict(train_df, test_df, attrs=['app'])
print('a')
df_os_channel     = predict(train_df, test_df, attrs=['os', 'channel'])
print('b')
df_app_os = predict(train_df, test_df, attrs=['app', 'os'])
print('c')
#df_app_chnnl_dev = predict(train_df, test_df, attrs=['app', 'channel', 'device'])

w_ac  = 0.2
w_a   = 0.2
w_oc = 0.25
x = 0.35
submit_df = (df_app_chnnl['is_attributed'] * w_ac + x * df_app_os['is_attributed'] + df_app['is_attributed'] * w_a + df_os_channel['is_attributed'] * w_oc).reset_index()
submit_df[['click_id', 'is_attributed']].to_csv('../output/subnew.csv', index=False)
print ('All done..')