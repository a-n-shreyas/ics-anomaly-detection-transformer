import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import as_strided


def create_windows_vectorized(df, window_size=50, stride=1):

    # 1. Isolate the feature and label data
    feature_columns = df.columns.difference(['Label'])
    features = df[feature_columns].values
    labels = df['Label'].values

    # 2. Calculate the shape of the output arrays
    num_windows = (features.shape[0] - window_size) // stride + 1
    num_features = features.shape[1]

    # 3. Create the feature windows (X)
    sub_shape = (window_size, num_features)
    view_shape = (num_windows,) + sub_shape
    byte_stride = features.strides[0] * stride
    feature_strides = (byte_stride,) + features.strides

    # --- CRUCIAL: This line defines X ---
    X = as_strided(features, shape=view_shape, strides=feature_strides)

    # 4. Create the label windows (y)
    label_view_shape = (num_windows, window_size)
    label_strides = (byte_stride,) + labels.strides
    label_windows = as_strided(labels, shape=label_view_shape, strides=label_strides)

    # --- CRUCIAL: This line defines y ---
    y = np.any(label_windows, axis=1).astype(int)

    print(f"✅ Vectorized windows created: {X.shape[0]} sequences of shape {X.shape[1:]}")

    return X, y
def clean_wadi_data(df):

    # Step 1: Combine 'Date' and 'Time' into a single 'Timestamp' column
    df['Timestamp'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'],
        format='%m/%d/%Y %I:%M:%S.%f %p',
        errors='coerce'
    )
    df.drop(columns=['Date', 'Time'], inplace=True)

    # Step 2: Clean column names
    df.columns = df.columns.str.replace(r'.*\\', '', regex=True).str.strip()

    # Step 3: Convert all feature columns to numeric data types
    columns_to_drop = ['Row', 'Timestamp']
    feature_cols = df.columns.drop(columns_to_drop, errors='ignore')
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')

    # Step 4: Handle any missing values on numeric columns only
    df[feature_cols] = df[feature_cols].ffill().bfill()

    # Step 5: Final cleanup
    df.drop(columns=['Row'], inplace=True, errors='ignore')
    df.set_index('Timestamp', inplace=True)

    return df