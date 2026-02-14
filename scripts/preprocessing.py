import os
import json
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(input_path):
    files = [f for f in os.listdir(input_path) if f.endswith('.csv')]
    frames = []
    for f in files:
        df = pd.read_csv(os.path.join(input_path, f))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def clean_data(df):
    initial_rows = len(df)
    df = df.dropna(thresh=int(len(df.columns) * 0.7))
    df = df.drop_duplicates()

    for col in df.select_dtypes(include=[np.number]).columns:
        q1 = df[col].quantile(0.01)
        q99 = df[col].quantile(0.99)
        df[col] = df[col].clip(q1, q99)

    print(f"Cleaned {initial_rows - len(df)} rows, {len(df)} remaining")
    return df


def encode_features(df, target_col):
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        if col == target_col:
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    return df, label_encoders


def normalize_features(df, target_col):
    feature_cols = [c for c in df.columns if c != target_col]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler


def split_and_save(df, output_path, target_col, test_size=0.2, val_size=0.1):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=42)

    for name, split in [('train', train_df), ('validation', val_df), ('test', test_df)]:
        split_dir = os.path.join(output_path, name)
        os.makedirs(split_dir, exist_ok=True)
        split.to_csv(os.path.join(split_dir, 'data.csv'), index=False)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-col', type=str, required=True)
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--val-size', type=float, default=0.1)
    args = parser.parse_args()

    input_path = '/opt/ml/processing/input'
    output_path = '/opt/ml/processing/output'

    df = load_data(input_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    df = clean_data(df)
    df, encoders = encode_features(df, args.target_col)
    df, scaler = normalize_features(df, args.target_col)
    split_and_save(df, output_path, args.target_col, args.test_size, args.val_size)

    metadata = {
        'input_rows': len(df),
        'features': len(df.columns) - 1,
        'target': args.target_col,
        'splits': {'test_size': args.test_size, 'val_size': args.val_size},
    }
    with open(os.path.join(output_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)


if __name__ == '__main__':
    main()
