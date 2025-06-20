# Bus Demand Forecasting Web API

This project predicts bus seat demand for specific routes and dates using historical booking and search data. It provides a machine learning pipeline and a FastAPI web API for real-time predictions.

## Features

- Data pipeline and feature engineering (Python scripts)
- RandomForestRegressor model
- FastAPI backend for predictions
- Example script to extract real feature values for API testing

## Requirements

- Python 3.8+
- pandas
- scikit-learn
- joblib
- fastapi
- uvicorn

Install requirements:

```bash
pip install pandas scikit-learn joblib fastapi uvicorn
```

## Usage

### 1. Train the Model and Save Artifacts

Run the main training script to process data, train the model, and save the model and encoders:

```bash
python main.py
```

This will generate `rf_model.joblib` and `encoders.joblib` in your project directory.

### 2. Start the FastAPI Server

Run the API server:

```bash
python -m uvicorn app:app --reload
```

The API will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000)

### 3. Test the API

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for Swagger UI. Use the `/predict` endpoint and fill in the request body with real feature values.

#### Example Request Body

```
{
  "doj": "2023-03-01",
  "srcid": "45",
  "destid": "46",
  "srcid_region": "Karnataka",
  "destid_region": "Tamil Nadu",
  "srcid_tier": "Tier 1",
  "destid_tier": "Tier 1",
  "cumsum_seatcount_15": 16.0,
  "cumsum_searchcount_15": 480.0,
  "cumsum_seatcount_10": 52.0,
  "cumsum_searchcount_10": 876.0,
  "cumsum_seatcount_7": 134.0,
  "cumsum_searchcount_7": 1434.0,
  "route_mean_seatcount": 4218.648809523809
}
```

#### Example Response

```
{
  "final_seatcount": 2806
}
```

### 4. Extract Real Feature Values

To get real feature values for a specific route/date, run:

```bash
python read_data.py
```

Copy the output and use it as input for the API.

## Project Structure

- `main.py` — Data processing, feature engineering, model training, and saving artifacts
- `app.py` — FastAPI backend for predictions
- `read_data.py` — Script to extract real feature values for API testing
- `train/` — Contains `train.csv` and `transactions.csv`
- `test.csv` — Test set for competition

## Notes

- Use only values for categorical features that were present in the training data.
- You can extend the feature set by editing `main.py` and retraining the model.

## License

This project is for educational and hackathon use only.
