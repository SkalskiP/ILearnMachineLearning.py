import pandas as pd

# File settings
INPUT_FILE = "../input/test.csv"
OUTPUT_FILE = "../input/test.h5"

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
    'click_id'
]

print('loading full data set...')
test_df = pd.read_csv(INPUT_FILE, parse_dates=['click_time'], dtype=dtype, usecols=usecols)
print('saving to file...')
test_df.to_hdf(OUTPUT_FILE, 'test_df', format ="fixed", mode="w")
