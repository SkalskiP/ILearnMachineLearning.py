# All credits go to original authors!
import pandas as pd

test_files = ['../../output/subnew.csv',
              '../../output/xgb_sub.csv',
              '../../output/sub_lgb_balanced99.csv',
              '../../output/dl_support.csv',
              '../../output/submission.csv']

model_test_data = []
for test_file in test_files:
    print('read ' + test_file)
    model_test_data.append(pd.read_csv(test_file, encoding='utf-8'))
n_models = len(model_test_data)

weights = [0.3*0.12, 0.3*0.23, 0.3*0.65, 0.3, 0.4]
column_name = 'is_attributed'

print('predict')
test_predict_column = [0.] * len(model_test_data[0][column_name])
for ind in range(0, n_models):
    test_predict_column += model_test_data[ind][column_name] * weights[ind]

print('make result')
final_result = model_test_data[0]['click_id']
final_result = pd.concat((final_result, pd.DataFrame(
    {column_name: test_predict_column})), axis=1)
final_result.to_csv("../../output/average_result.csv", index=False)