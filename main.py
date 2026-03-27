import pandas as pd
import numpy as np

# adjust this number to modify what % of each dataset to use.
# intermediate X matrix, at samplesize = 0.5 requires 70GB+
# reduce this number when RAM is limited
SAMPLE_SIZE = 0.05


def create_np_data():
    # generate filelist for yellow cab parquet data
    files = ["data/yellow_tripdata_2024-12.parquet"]
    for month in range(1, 12):
        files.append(f"data/yellow_tripdata_2025-{month:02d}.parquet")

    # load data
    df_list = []
    for f in files:
        temp_df = pd.read_parquet(f, engine="pyarrow")
        temp_df = temp_df.sample(frac=SAMPLE_SIZE, random_state=100)
        df_list.append(temp_df)

    df = pd.concat(df_list, ignore_index=True)

    print(f"Loaded {len(df):,} rows across {len(files):,} months.")

    # https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf

    # compute column for trip time (in min)
    df["trip_time_mins"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    # hour of day (related to congestion/demand)
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour

    # day of week
    df["day_of_week"] = df["tpep_pickup_datetime"].dt.dayofweek

    # filter RateCodeID for just standard rates, not Newark/JFK special rates
    df = df[df["RatecodeID"] == 1]

    # price before tip
    df["price_before_tip"] = df["total_amount"] - df["tip_amount"]

    # month
    df["month"] = df["tpep_pickup_datetime"].dt.month

    print(f"Processed {len(df):,} rows ready, across {len(files):,} months.")

    """
    Structure of the X matrix (Xw = y)
    PULocationID (one-hot encoded)
    DOLocationID (one-hot encoded)
    pickup_hour (one-hot encoded)
    day_of_week (one-hot encoded)
    month (one-hot-encoded)
    
    Dropped columns: column 0 from each of the four one-hot encoded
    
    Structure of the y vector
    price_before_tip
    """

    pu_dummies = pd.get_dummies(df["PULocationID"], prefix="PU", drop_first=True)
    do_dummies = pd.get_dummies(df["DOLocationID"], prefix="DO", drop_first=True)
    hour_dummies = pd.get_dummies(df["pickup_hour"], prefix="H", drop_first=True)
    dow_dummies = pd.get_dummies(df["day_of_week"], prefix="DOW", drop_first=True)
    month_dummies = pd.get_dummies(df["month"], prefix="MONTH", drop_first=True)

    # store the hot-encode columns for use later
    pu_cols = list(pu_dummies.columns)
    do_cols = list(do_dummies.columns)
    h_cols = list(hour_dummies.columns)
    dow_cols = list(dow_dummies.columns)
    month_cols = list(month_dummies.columns)

    pd.Series(pu_cols).to_csv("cols_pu.csv", index=False, header=False)
    pd.Series(do_cols).to_csv("cols_do.csv", index=False, header=False)
    pd.Series(h_cols).to_csv("cols_h.csv", index=False, header=False)
    pd.Series(dow_cols).to_csv("cols_dow.csv", index=False, header=False)
    pd.Series(month_cols).to_csv("cols_month.csv", index=False, header=False)

    X = np.hstack(
        [
            pu_dummies.values.astype(float),
            do_dummies.values.astype(float),
            hour_dummies.values.astype(float),
            dow_dummies.values.astype(float),
            month_dummies.values.astype(float),
        ]
    )

    y = df["price_before_tip"].values

    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")

    w = np.linalg.inv(X.T @ X) @ X.T @ y

    np.save("W_vector.npy", w)

    # print(f"Gram Matrix Rank: {np.linalg.matrix_rank(XTX)}")
    print(f"w vector shape: {w.shape}")

    w_np, *_ = np.linalg.lstsq(X, y, rcond=None)
    print(f"Numpy Check (Verfication): {np.allclose(w, w_np)}")


def validate_model(test_file):
    # load pandas columns
    train_pu_cols = pd.read_csv("cols_pu.csv", header=None).iloc[:, 0].tolist()
    train_do_cols = pd.read_csv("cols_do.csv", header=None).iloc[:, 0].tolist()
    train_h_cols = pd.read_csv("cols_h.csv", header=None).iloc[:, 0].tolist()
    train_dow_cols = pd.read_csv("cols_dow.csv", header=None).iloc[:, 0].tolist()
    train_month_cols = pd.read_csv("cols_month.csv", header=None).iloc[:, 0].tolist()

    w = np.load("W_vector.npy")

    df = pd.read_parquet(test_file, engine="pyarrow")
    df = df[df["RatecodeID"] == 1]
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["day_of_week"] = df["tpep_pickup_datetime"].dt.dayofweek
    df["price_before_tip"] = df["total_amount"] - df["tip_amount"]
    df["month"] = df["tpep_pickup_datetime"].dt.month

    # one hot encode the expected columns, reindex for order
    pu_dummies = pd.get_dummies(df["PULocationID"], prefix="PU").reindex(
        columns=train_pu_cols, fill_value=0
    )
    do_dummies = pd.get_dummies(df["DOLocationID"], prefix="DO").reindex(
        columns=train_do_cols, fill_value=0
    )
    hour_dummies = pd.get_dummies(df["pickup_hour"], prefix="H").reindex(
        columns=train_h_cols, fill_value=0
    )
    dow_dummies = pd.get_dummies(df["day_of_week"], prefix="DOW").reindex(
        columns=train_dow_cols, fill_value=0
    )
    month_dummies = pd.get_dummies(df["month"], prefix="MONTH").reindex(
        columns=train_month_cols, fill_value=0
    )

    # X_test matrix
    X_test = np.hstack(
        [
            pu_dummies.values.astype(float),
            do_dummies.values.astype(float),
            hour_dummies.values.astype(float),
            dow_dummies.values.astype(float),
            month_dummies.values.astype(float),
        ]
    )

    # y_actual for actual fares
    y_actual = df["price_before_tip"].values

    # y_pred for predicted fares
    y_pred = X_test @ w

    # performance

    # mean absolute error
    mae = np.mean(np.abs(y_actual - y_pred))

    # mean squared error (MSE)
    mse = np.mean((y_actual - y_pred) ** 2)

    # residual sum of squares
    ss_res = np.sum((y_actual - y_pred) ** 2)

    # total sum of squares
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)

    r2 = 1 - (ss_res / ss_tot)

    print(f"\nresults for {test_file}")
    print(f"mean absolute error (MAE): ${mae:.2f}")
    print(f"mean squared error (MSE): {mse:.2f}")
    print(f"R-squared (R^2): {r2:.4f}")


def predict_fare(pu_id, do_id, pickup_hour, day_of_week, month):
    import numpy as np
    import pandas as pd

    # load training column structure
    train_pu_cols = pd.read_csv("cols_pu.csv", header=None).iloc[:, 0].tolist()
    train_do_cols = pd.read_csv("cols_do.csv", header=None).iloc[:, 0].tolist()
    train_h_cols = pd.read_csv("cols_h.csv", header=None).iloc[:, 0].tolist()
    train_dow_cols = pd.read_csv("cols_dow.csv", header=None).iloc[:, 0].tolist()
    train_month_cols = pd.read_csv("cols_month.csv", header=None).iloc[:, 0].tolist()

    # load weights
    w = np.load("W_vector.npy")

    # create single-row df
    df = pd.DataFrame(
        {
            "PULocationID": [pu_id],
            "DOLocationID": [do_id],
            "pickup_hour": [pickup_hour],
            "day_of_week": [day_of_week],
            "month": [month],
        }
    )

    # one-hot encode, align cols w/ pandas csvs
    pu_dummies = pd.get_dummies(df["PULocationID"], prefix="PU").reindex(
        columns=train_pu_cols, fill_value=0
    )
    do_dummies = pd.get_dummies(df["DOLocationID"], prefix="DO").reindex(
        columns=train_do_cols, fill_value=0
    )
    hour_dummies = pd.get_dummies(df["pickup_hour"], prefix="H").reindex(
        columns=train_h_cols, fill_value=0
    )
    dow_dummies = pd.get_dummies(df["day_of_week"], prefix="DOW").reindex(
        columns=train_dow_cols, fill_value=0
    )
    month_dummies = pd.get_dummies(df["month"], prefix="MONTH").reindex(
        columns=train_month_cols, fill_value=0
    )

    # build single feature vector
    X = np.hstack(
        [
            pu_dummies.values,
            do_dummies.values,
            hour_dummies.values,
            dow_dummies.values,
            month_dummies.values,
        ]
    ).astype(float)

    # prediction
    y_pred = X @ w

    return float(y_pred[0])


# create_np_data()

# validate_model("test/yellow_tripdata_2024-05.parquet")

# print(predict_fare(202, 145, 23, 2, 3))
# A trip from R.I. to Gantry Plaza S.P. at 11PM on a Wednesday in March, predicted to be $23 in fare

print(predict_fare(202, 13, 20, 3, 3))
# A trip from R.I. to TriBeCa at 8PM on a Thursday in March, predicted to be $18.90 in fare (actual: $60)

print(predict_fare(202, 145, 20, 3, 3))
# A trip from R.I. to TriBeCa at 8PM on a Thursday in March, predicted to be $18.29 in fare (actual: $17.93)
