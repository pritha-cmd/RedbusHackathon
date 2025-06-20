import pandas as pd

# File paths
TRAIN_PATH = 'train/train/train.csv'
TRANSACTIONS_PATH = 'train/train/transactions.csv'

# Load data
train = pd.read_csv(TRAIN_PATH)
transactions = pd.read_csv(TRANSACTIONS_PATH)

# Choose a specific example
doj = '2023-03-01'
srcid = '45'
destid = '46'

# Get lag features for dbd=15, 10, 7
tx_15 = transactions[(transactions['doj'] == doj) & (transactions['srcid'].astype(str) == srcid) & (transactions['destid'].astype(str) == destid) & (transactions['dbd'] == 15)]
tx_10 = transactions[(transactions['doj'] == doj) & (transactions['srcid'].astype(str) == srcid) & (transactions['destid'].astype(str) == destid) & (transactions['dbd'] == 10)]
tx_7 = transactions[(transactions['doj'] == doj) & (transactions['srcid'].astype(str) == srcid) & (transactions['destid'].astype(str) == destid) & (transactions['dbd'] == 7)]

# Get region/tier info from tx_15 (should be the same for all dbd)
if not tx_15.empty:
    srcid_region = tx_15.iloc[0]['srcid_region']
    destid_region = tx_15.iloc[0]['destid_region']
    srcid_tier = tx_15.iloc[0]['srcid_tier']
    destid_tier = tx_15.iloc[0]['destid_tier']
else:
    srcid_region = destid_region = srcid_tier = destid_tier = None

# Get lag values (use .iloc[0] if not empty, else None)
def get_lag(df, seat_col, search_col):
    if not df.empty:
        return df.iloc[0][seat_col], df.iloc[0][search_col]
    return None, None

cumsum_seatcount_15, cumsum_searchcount_15 = get_lag(tx_15, 'cumsum_seatcount', 'cumsum_searchcount')
cumsum_seatcount_10, cumsum_searchcount_10 = get_lag(tx_10, 'cumsum_seatcount', 'cumsum_searchcount')
cumsum_seatcount_7, cumsum_searchcount_7 = get_lag(tx_7, 'cumsum_seatcount', 'cumsum_searchcount')

# Route mean seatcount from train
group = train[(train['srcid'].astype(str) == srcid) & (train['destid'].astype(str) == destid)]
route_mean_seatcount = group['final_seatcount'].mean() if not group.empty else None

# Print all values for FastAPI input
print({
    'doj': doj,
    'srcid': srcid,
    'destid': destid,
    'srcid_region': srcid_region,
    'destid_region': destid_region,
    'srcid_tier': srcid_tier,
    'destid_tier': destid_tier,
    'cumsum_seatcount_15': cumsum_seatcount_15,
    'cumsum_searchcount_15': cumsum_searchcount_15,
    'cumsum_seatcount_10': cumsum_seatcount_10,
    'cumsum_searchcount_10': cumsum_searchcount_10,
    'cumsum_seatcount_7': cumsum_seatcount_7,
    'cumsum_searchcount_7': cumsum_searchcount_7,
    'route_mean_seatcount': route_mean_seatcount
})

  