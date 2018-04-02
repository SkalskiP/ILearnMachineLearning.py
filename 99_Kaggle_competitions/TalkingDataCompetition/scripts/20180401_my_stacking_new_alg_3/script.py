LOGIT_WEIGHT = .8

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.special import expit, logit

almost_zero = 1e-10
almost_one = 1 - almost_zero

models = {
  'xgb  ':  "../../output/sub_xgb_hist_R_50m.csv",
  """'nn   ':  "../201803301_deep_learning_support_imbalance_architect_9674/imbalanced_data.csv","""
  'lgb  ':  "../20180330_try_pranav_s_r_lgbm_in_python/sub_lgbm_r_to_python_withcv.csv",
  'lgb2 ':  "../20180401_my_lightgbm_6/sub_lgbm_r_to_python_withcv.csv"
  }
  
weights = {
  'xgb  ':  .25,    #0.9686
  """'nn   ':  .15,    #0.9662"""
  'lgb  ':  .60,    #0.9694
  'lgb2 ':  .15     #0.9684
  }
  
print (sum(weights.values()))


subs = {m:pd.read_csv(models[m]) for m in models}
first_model = list(models.keys())[0]
n = subs[first_model].shape[0]

ranks = {s:subs[s]['is_attributed'].rank()/n for s in subs}
logits = {s:subs[s]['is_attributed'].clip(almost_zero,almost_one).apply(logit) for s in subs}

logit_avg = 0
rank_avg = 0
for m in models:
    s = logits[m].std()
    print(m, s)
    logit_avg = logit_avg + weights[m]*logits[m] / s
    rank_avg = rank_avg + weights[m]*ranks[m]

logit_rank_avg = logit_avg.rank()/n
final_avg = LOGIT_WEIGHT*logit_rank_avg + (1-LOGIT_WEIGHT)*rank_avg

final_sub = pd.DataFrame()
final_sub['click_id'] = subs[first_model]['click_id']
final_sub['is_attributed'] = final_avg

print( final_sub.head() )

final_sub.to_csv("my_mix_20190402.csv", index=False, float_format='%.8f')
