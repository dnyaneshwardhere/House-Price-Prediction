import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings("ignore")

class DataLoader:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path

    def load_data(self):
        train = pd.read_csv(self.train_path)
        test = pd.read_csv(self.test_path)
        return train, test

class Preprocessor:
    def __init__(self, df):
        self.df = df

    def handle_missing(self):
        self.df = self.df.fillna(self.df.median(numeric_only=True))
        self.df = self.df.fillna("None")
        return self.df

    def encode(self):
        for col in self.df.select_dtypes(include='object').columns:
            self.df[col] = LabelEncoder().fit_transform(self.df[col].astype(str))
        return self.df

    def scale(self):
        scaler = StandardScaler()
        num_cols = self.df.select_dtypes(include=np.number).columns
        self.df[num_cols] = scaler.fit_transform(self.df[num_cols])
        return self.df

class ModelTrainer:
    def __init__(self, model):
        self.model = model

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate_regression(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        print("MSE:", mse)
        print("RMSE:", rmse)
        print("R² Score:", r2)
        return mse, rmse, r2  

def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def save_predictions(ids, preds, filepath):
    df = pd.DataFrame({"Id": ids, "SalePrice": preds})
    df.to_csv(filepath, index=False)

def main():
    path = "E:/Internships & Certificates/Celebal/Celebal/Assignment 5"
    train_file = os.path.join(path, "train.csv")
    test_file = os.path.join(path, "test.csv")
    sub_file = os.path.join(path, "sample_submission.csv")

    loader = DataLoader(train_file, test_file)
    train_df, test_df = loader.load_data()

    y = train_df["SalePrice"]
    train_df.drop(["SalePrice"], axis=1, inplace=True)

    all_data = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    pre = Preprocessor(all_data)
    data = pre.handle_missing()
    data = pre.encode()
    data = pre.scale()

    X_train = data.iloc[:len(y), :]
    X_test = data.iloc[len(y):, :]

    X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y, test_size=0.2, random_state=42)

    model = ModelTrainer(Ridge(alpha=1.0))
    model.train(X_train_sub, y_train_sub)

    val_preds = model.predict(X_val)
    mse, rmse, r2 = model.evaluate_regression(y_val, val_preds)

    # 1. Scatter plot
    plt.figure(figsize=(6, 4))
    plt.scatter(y_val, val_preds, alpha=0.5, c='blue')
    plt.xlabel("Actual SalePrice")
    plt.ylabel("Predicted SalePrice")
    plt.title("Validation Predictions")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(path, "validation_predictions.png"))
    plt.close()

    # 2. Cross-validation
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(Ridge(alpha=1.0), X_train, y, scoring="r2", cv=5)
    print("Cross-Validation R² Scores:", scores)
    print("Average CV R² Score:", scores.mean())

    coef = model.model.coef_
    feature_names = X_train.columns
    top_indices = np.argsort(np.abs(coef))[::-1][:15]
    top_features = feature_names[top_indices]
    top_values = coef[top_indices]

    plt.figure(figsize=(8, 5))
    plt.barh(top_features, top_values)
    plt.xlabel("Ridge Coefficient")
    plt.title("Top 15 Feature Importances")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(path, "ridge_feature_importances.png"))
    plt.close()

    # 3. Final predictions
    test_preds = model.predict(X_test)
    save_predictions(test_df["Id"], test_preds, os.path.join(path, "my_submission.csv"))

    # 4. Save evaluation results to JSON
    eval_metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "R2_Score": r2,
        "CrossVal_R2_Avg": scores.mean()
    }
    save_json(eval_metrics, os.path.join(path, "evaluation_metrics.json"))

if __name__ == "__main__":
    main()
