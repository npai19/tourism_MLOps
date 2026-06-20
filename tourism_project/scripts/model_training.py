import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import joblib
from huggingface_hub import hf_hub_download, create_repo, upload_file
import os
import sys

# Retrieve environment variables
HF_TOKEN = os.environ.get('HF_TOKEN')
HF_DATASET_REPO = os.environ.get('HF_DATASET_REPO') # e.g., 'your-username/tourism_dataset'
HF_MODEL_REPO = os.environ.get('HF_MODEL_REPO') # e.g., 'your-username/tourism_prediction_decision_tree_model'

if not HF_TOKEN or not HF_DATASET_REPO or not HF_MODEL_REPO:
    print("Error: HF_TOKEN, HF_DATASET_REPO, and HF_MODEL_REPO environment variables must be set.")
    sys.exit(1)

# Define local paths for downloaded data and saved model
LOCAL_TRAIN_FILE = 'tourism_train.csv'
LOCAL_TEST_FILE = 'tourism_test.csv'
LOCAL_MODEL_PATH = 'tourism_project/model_building/best_decision_tree_model.joblib'

# Ensure model_building directory exists
os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)

# Allow MLflow to use the filesystem backend
os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'

# Set the MLflow tracking URI to a local directory for now (will be uploaded as artifact)
mlflow.set_tracking_uri("file:./tourism_project/mlruns")
mlflow.set_experiment("Tourism_Prediction_DecisionTree_Experiment")

print(f"Downloading {HF_DATASET_REPO}/{LOCAL_TRAIN_FILE}...")
# Download the training dataset from Hugging Face
train_csv_path = hf_hub_download(
    repo_id=HF_DATASET_REPO,
    filename=LOCAL_TRAIN_FILE,
    repo_type='dataset',
    token=HF_TOKEN
)
df_train_loaded = pd.read_csv(train_csv_path)
print(f"Downloaded {LOCAL_TRAIN_FILE}. Shape: {df_train_loaded.shape}")

print(f"Downloading {HF_DATASET_REPO}/{LOCAL_TEST_FILE}...")
# Download the testing dataset from Hugging Face
test_csv_path = hf_hub_download(
    repo_id=HF_DATASET_REPO,
    filename=LOCAL_TEST_FILE,
    repo_type='dataset',
    token=HF_TOKEN
)
df_test_loaded = pd.read_csv(test_csv_path)
print(f"Downloaded {LOCAL_TEST_FILE}. Shape: {df_test_loaded.shape}")

# Separate features (X) and target (y) for training and testing sets
X_train = df_train_loaded.drop('ProdTaken', axis=1)
y_train = df_train_loaded['ProdTaken']
X_test = df_test_loaded.drop('ProdTaken', axis=1)
y_test = df_test_loaded['ProdTaken']
print("Features and target separated.")

with mlflow.start_run():
    mlflow.log_param("model_type", "DecisionTreeClassifier")

    dt_classifier = DecisionTreeClassifier(random_state=42)
    param_grid = {
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    print("Starting GridSearchCV for hyperparameter tuning...")
    grid_search = GridSearchCV(dt_classifier, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_dt_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print("Best parameters found:", best_params)

    mlflow.log_params(best_params)

    y_pred = best_dt_model.predict(X_test)
    y_pred_proba = best_dt_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    })

    mlflow.sklearn.log_model(best_dt_model, "decision_tree_model")
    print("Model logged to MLflow.")

    # Save the best model locally
    joblib.dump(best_dt_model, LOCAL_MODEL_PATH)
    print(f"Best model saved locally to: {LOCAL_MODEL_PATH}")

    # Upload the best model to Hugging Face Model Hub
    create_repo(repo_id=HF_MODEL_REPO, repo_type='model', token=HF_TOKEN, exist_ok=True)
    upload_file(
        path_or_fileobj=LOCAL_MODEL_PATH,
        path_in_repo='best_decision_tree_model.joblib',
        repo_id=HF_MODEL_REPO,
        repo_type='model',
        token=HF_TOKEN
    )
    print(f"Uploaded {LOCAL_MODEL_PATH} to {HF_MODEL_REPO}/best_decision_tree_model.joblib")

print("Model training complete and model uploaded.")
