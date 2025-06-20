from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI()

# Load model and encoders
script_dir = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(script_dir, 'rf_model.joblib'))
encoders = joblib.load(os.path.join(script_dir, 'encoders.joblib'))

# Define input schema
class PredictionRequest(BaseModel):
    doj: str
    srcid: str
    destid: str
    srcid_region: str
    destid_region: str
    srcid_tier: str
    destid_tier: str
    cumsum_seatcount_15: float
    cumsum_searchcount_15: float
    cumsum_seatcount_10: float
    cumsum_searchcount_10: float
    cumsum_seatcount_7: float
    cumsum_searchcount_7: float
    route_mean_seatcount: float

@app.post('/predict')
def predict(req: PredictionRequest):
    # Feature engineering
    df = pd.DataFrame([req.dict()])
    df['doj'] = pd.to_datetime(df['doj'])
    df['dow'] = df['doj'].dt.dayofweek
    df['month'] = df['doj'].dt.month
    df['is_weekend'] = df['dow'].isin([5, 6]).astype(int)
    df['day'] = df['doj'].dt.day

    # Encode categorical features
    cat_cols = ['srcid_region', 'destid_region', 'srcid_tier', 'destid_tier']
    for col in cat_cols:
        le = encoders[col]
        df[col] = le.transform(df[col].astype(str))

    # Select features in the right order
    feature_cols = [
        'cumsum_seatcount_15', 'cumsum_searchcount_15',
        'cumsum_seatcount_10', 'cumsum_searchcount_10',
        'cumsum_seatcount_7', 'cumsum_searchcount_7',
        'srcid_region', 'destid_region', 'srcid_tier', 'destid_tier',
        'dow', 'month', 'is_weekend', 'day', 'route_mean_seatcount'
    ]
    X = df[feature_cols]
    pred = model.predict(X)[0]
    return {'final_seatcount': int(round(pred))} 