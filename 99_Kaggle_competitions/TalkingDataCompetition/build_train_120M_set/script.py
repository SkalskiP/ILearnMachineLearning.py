import pandas as pd

# File settings
INPUT_FILE = "../input/train.csv"
OUTPUT_FILE = "../input/train120M.h5"

TOTAL_ROWS = 184903891-1
CHUNK_SIZE = 120000000
START_POINT = TOTAL_ROWS - CHUNK_SIZE

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
train_df = pd.read_csv(INPUT_FILE, parse_dates=['click_time'], dtype=dtype, skiprows=range(1, START_POINT), nrows=CHUNK_SIZE, usecols=usecols)
print('saving to file...')
train_df.to_hdf(OUTPUT_FILE, 'train_df_120', format ="fixed", mode="w")
