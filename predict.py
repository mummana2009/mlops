import sys
import pickle
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer

def load_model():
    with open("model.bin", "rb") as f_in:
        dv, model = pickle.load(f_in)
    return dv, model

def load_data(year, month):
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    df = pd.read_parquet(url)
    df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    df = df[(df['duration'] >= 1) & (df['duration'] <= 60)]
    df['PULocationID'] = df['PULocationID'].astype(str)
    df['DOLocationID'] = df['DOLocationID'].astype(str)
    return df

def predict(df, dv, model):
    dicts = df[['PULocationID', 'DOLocationID']].to_dict(orient='records')
    X = dv.transform(dicts)
    y_pred = model.predict(X)
    return np.mean(y_pred)

def main():
    year = int(sys.argv[1])
    month = int(sys.argv[2])

    dv, model = load_model()
    df = load_data(year, month)

    mean_duration = predict(df, dv, model)
    print(f"✅ Mean predicted duration for {month:02d}/{year}: {mean_duration:.2f} minutes")

if __name__ == "__main__":
    main()
