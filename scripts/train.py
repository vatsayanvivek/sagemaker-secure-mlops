import os
import json
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def load_channel(channel_name):
    path = os.path.join('/opt/ml/input/data', channel_name)
    files = [f for f in os.listdir(path) if f.endswith('.csv')]
    frames = [pd.read_csv(os.path.join(path, f)) for f in files]
    return pd.concat(frames, ignore_index=True)


def train_model(train_df, val_df, target_col, hyperparams):
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]

    model = GradientBoostingClassifier(
        n_estimators=hyperparams.get('n_estimators', 200),
        learning_rate=hyperparams.get('learning_rate', 0.01),
        max_depth=hyperparams.get('max_depth', 6),
        min_samples_split=hyperparams.get('min_samples_split', 10),
        subsample=hyperparams.get('subsample', 0.8),
        random_state=42,
    )

    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    val_proba = model.predict_proba(X_val)[:, 1] if len(np.unique(y_val)) == 2 else None

    metrics = {
        'accuracy': accuracy_score(y_val, val_preds),
        'precision': precision_score(y_val, val_preds, average='weighted'),
        'recall': recall_score(y_val, val_preds, average='weighted'),
        'f1': f1_score(y_val, val_preds, average='weighted'),
    }

    if val_proba is not None:
        metrics['auc'] = roc_auc_score(y_val, val_proba)

    return model, metrics


def save_model(model, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, 'model.joblib'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_depth', type=int, default=6)
    parser.add_argument('--target-col', type=str, default='target')
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    args = parser.parse_args()

    train_df = load_channel('train')
    val_df = load_channel('validation')

    print(f"Training data: {train_df.shape}, Validation data: {val_df.shape}")

    hyperparams = {
        'n_estimators': args.epochs,
        'learning_rate': args.learning_rate,
        'max_depth': args.max_depth,
    }

    model, metrics = train_model(train_df, val_df, args.target_col, hyperparams)

    print("Validation metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    save_model(model, args.model_dir)

    metrics_path = os.path.join(args.model_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Model saved to {args.model_dir}")


if __name__ == '__main__':
    main()
