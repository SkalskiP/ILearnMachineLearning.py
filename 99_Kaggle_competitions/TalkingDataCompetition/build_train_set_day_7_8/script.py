import pandas as pd

# File settings
INPUT_FILE = "../input/train.csv"
OUTPUT_FILE = "../input/train78day.h5"

# Date settings
START_DATE = "2017-11-07 00:00:00"
END_DATE = "2017-11-08 23:59:59"

# Columns types
dtype = {
    'ip'            : 'uint32',
    'app'           : 'uint16',
    'device'        : 'uint16',
    'os'            : 'uint16',
    'channel'       : 'uint16',
    'is_attributed' : 'uint8',
    'click_id'      : 'uint32',
}

# Column names
usecols = [
    'ip',
    'app',
    'device',
    'os',
    'channel',
    'click_time',
    'is_attributed'
]

print('loading full data set...')
full_df = pd.read_csv(INPUT_FILE, parse_dates=['click_time'], dtype=dtype, usecols=usecols)
print('filtering by date...')
valid_df = full_df.loc[(full_df['click_time'] < END_DATE) & (full_df['click_time'] > START_DATE)]
print('saving to file...')
valid_df.to_hdf(OUTPUT_FILE, 'df_day_78', format ="fixed", mode="w")
