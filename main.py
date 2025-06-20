import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib
import os

# File paths
TRAIN_PATH = 'train/train/train.csv'
TRANSACTIONS_PATH = 'train/train/transactions.csv'
TEST_PATH = 'test.csv'

# Load data
print('Loading data...')
train = pd.read_csv(TRAIN_PATH)
transactions = pd.read_csv(TRANSACTIONS_PATH)
test = pd.read_csv(TEST_PATH)

print('Train shape:', train.shape)
print('Transactions shape:', transactions.shape)
print('Test shape:', test.shape)

# --- Feature Engineering ---
# Date features
def add_date_features(df, date_col):
    df[date_col] = pd.to_datetime(df[date_col])
    df['dow'] = df[date_col].dt.dayofweek  # 0=Monday
    df['month'] = df[date_col].dt.month
    df['is_weekend'] = df['dow'].isin([5, 6]).astype(int)
    df['day'] = df[date_col].dt.day
    return df

train = add_date_features(train, 'doj')
test = add_date_features(test, 'doj')

# Route popularity (mean seat count per route)
route_pop = train.groupby(['srcid', 'destid'])['final_seatcount'].mean().reset_index()
route_pop = route_pop.rename(columns={'final_seatcount': 'route_mean_seatcount'})
train = pd.merge(train, route_pop, on=['srcid', 'destid'], how='left')
test = pd.merge(test, route_pop, on=['srcid', 'destid'], how='left')

# Extract features for dbd == 15, 10, 7
tx_15 = transactions[transactions['dbd'] == 15].copy()
tx_10 = transactions[transactions['dbd'] == 10].copy()
tx_7 = transactions[transactions['dbd'] == 7].copy()

merge_cols = [
    'doj', 'srcid', 'destid',
    'cumsum_seatcount', 'cumsum_searchcount',
    'srcid_region', 'destid_region', 'srcid_tier', 'destid_tier'
]
tx_15 = pd.DataFrame(tx_15[merge_cols]).rename(columns={
    'cumsum_seatcount': 'cumsum_seatcount_15',
    'cumsum_searchcount': 'cumsum_searchcount_15'
})
tx_10 = pd.DataFrame(tx_10[['doj', 'srcid', 'destid', 'cumsum_seatcount', 'cumsum_searchcount']]).rename(columns={
    'cumsum_seatcount': 'cumsum_seatcount_10',
    'cumsum_searchcount': 'cumsum_searchcount_10'
})
tx_7 = pd.DataFrame(tx_7[['doj', 'srcid', 'destid', 'cumsum_seatcount', 'cumsum_searchcount']]).rename(columns={
    'cumsum_seatcount': 'cumsum_seatcount_7',
    'cumsum_searchcount': 'cumsum_searchcount_7'
})

# Ensure merge keys are all strings
def ensure_str(df):
    df['doj'] = df['doj'].astype(str)
    df['srcid'] = df['srcid'].astype(str)
    df['destid'] = df['destid'].astype(str)
    return df

train = ensure_str(train)
test = ensure_str(test)
tx_15 = ensure_str(tx_15)
tx_10 = ensure_str(tx_10)
tx_7 = ensure_str(tx_7)

# Assert DataFrame types
assert isinstance(train, pd.DataFrame), f"train is not a DataFrame, got {type(train)}"
assert isinstance(test, pd.DataFrame), f"test is not a DataFrame, got {type(test)}"
assert isinstance(tx_15, pd.DataFrame), f"tx_15 is not a DataFrame, got {type(tx_15)}"

# Debug: print columns and dtypes
print('train columns:', train.columns)
print('test columns:', test.columns)
print('tx_15 columns:', tx_15.columns)
print('train dtypes:\n', train.dtypes)
print('test dtypes:\n', test.dtypes)
print('tx_15 dtypes:\n', tx_15.dtypes)

# Merge features into train and test
train_merged = pd.merge(train, tx_15, on=['doj', 'srcid', 'destid'], how='left')
train_merged = pd.merge(train_merged, tx_10, on=['doj', 'srcid', 'destid'], how='left')
train_merged = pd.merge(train_merged, tx_7, on=['doj', 'srcid', 'destid'], how='left')
test_merged = pd.merge(test, tx_15, on=['doj', 'srcid', 'destid'], how='left')
test_merged = pd.merge(test_merged, tx_10, on=['doj', 'srcid', 'destid'], how='left')
test_merged = pd.merge(test_merged, tx_7, on=['doj', 'srcid', 'destid'], how='left')

print('\nMerged train shape:', train_merged.shape)
print('Merged test shape:', test_merged.shape)

# --- Baseline Model with More Features ---
feature_cols = [
    'cumsum_seatcount_15', 'cumsum_searchcount_15',
    'cumsum_seatcount_10', 'cumsum_searchcount_10',
    'cumsum_seatcount_7', 'cumsum_searchcount_7',
    'srcid_region', 'destid_region', 'srcid_tier', 'destid_tier',
    'dow', 'month', 'is_weekend', 'day', 'route_mean_seatcount'
]

# Drop rows with missing feature values in train
train_model = train_merged.dropna(subset=feature_cols + ['final_seatcount'])
X = train_model[feature_cols].copy()
y = train_model['final_seatcount']

# Encode categorical features
cat_cols = ['srcid_region', 'destid_region', 'srcid_tier', 'destid_tier']
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# --- Model Evaluation on Training Data ---
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print('\nTraining RandomForestRegressor on training split...')
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Predict on validation set
val_preds = model.predict(X_val)
rmse = mean_squared_error(y_val, val_preds) ** 0.5
print(f'Validation RMSE: {rmse:.4f}')

# Prepare test features
X_test = test_merged[feature_cols].copy()
for col in cat_cols:
    le = encoders[col]
    X_test[col] = le.transform(X_test[col].astype(str))

# Retrain on full data for final test prediction
model.fit(X, y)

# Predict on test set
print('Predicting on test set...')
test_preds = model.predict(X_test)

# Prepare submission
submission = pd.DataFrame({
    'route_key': test['route_key'],
    'final_seatcount': np.round(test_preds).astype(int)
})
submission.to_csv('submission_file.csv', index=False)
print('\nSubmission file saved as submission_file.csv')

# Save model and encoders for FastAPI
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'rf_model.joblib')
encoders_path = os.path.join(script_dir, 'encoders.joblib')
joblib.dump(model, model_path)
joblib.dump(encoders, encoders_path)
print('Model and encoders saved for FastAPI.') 