import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import create_repo, upload_file, hf_hub_download
import os
import sys

# Retrieve environment variables
HF_TOKEN = os.environ.get('HF_TOKEN')
HF_DATASET_REPO = os.environ.get('HF_DATASET_REPO') # e.g., 'your-username/tourism_dataset'

if not HF_TOKEN or not HF_DATASET_REPO:
    print("Error: HF_TOKEN and HF_DATASET_REPO environment variables must be set.")
    sys.exit(1)

# Define local paths
LOCAL_RAW_DATA_PATH = 'tourism_project/data/tourism.csv'
LOCAL_TRAIN_PATH = 'tourism_project/data/tourism_train.csv'
LOCAL_TEST_PATH = 'tourism_project/data/tourism_test.csv'

# Ensure local data directory exists
os.makedirs(os.path.dirname(LOCAL_TRAIN_PATH), exist_ok=True)

# 1. Download the raw dataset from Hugging Face
print(f"Downloading {HF_DATASET_REPO}/tourism.csv...")
raw_csv_path = hf_hub_download(
    repo_id=HF_DATASET_REPO,
    filename='tourism.csv',
    repo_type='dataset',
    token=HF_TOKEN
)
df = pd.read_csv(raw_csv_path)
print(f"Raw dataset loaded. Shape: {df.shape}")

# 2. Initial Data Cleaning
df_cleaned = df.drop(columns=['Unnamed: 0', 'CustomerID', 'Designation'], errors='ignore')
print(f"DataFrame shape after dropping columns: {df_cleaned.shape}")

# Impute missing values for numerical columns
for col in ['Age', 'MonthlyIncome', 'DurationOfPitch', 'PreferredPropertyStar', 'NumberOfChildrenVisiting']:
    if col in df_cleaned.columns:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

# Impute missing values for categorical columns
for col in ['TypeofContact', 'MaritalStatus', 'Gender']:
    if col in df_cleaned.columns:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
print("Missing values handled.")

# 3. Encoding Categorical Features
categorical_cols = df_cleaned.select_dtypes(include=['object']).columns.tolist()
df_encoded = pd.get_dummies(df_cleaned, columns=categorical_cols, drop_first=True)
print(f"DataFrame shape after encoding categorical features: {df_encoded.shape}")

# 4. Split the dataset
X = df_encoded.drop('ProdTaken', axis=1)
y = df_encoded['ProdTaken']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

df_train = pd.concat([X_train, y_train], axis=1)
df_test = pd.concat([X_test, y_test], axis=1)
print(f"Training set shape: {df_train.shape}")
print(f"Testing set shape: {df_test.shape}")

# 5. Save the training and testing dataframes locally
df_train.to_csv(LOCAL_TRAIN_PATH, index=False)
df_test.to_csv(LOCAL_TEST_PATH, index=False)
print(f"Saved training data to: {LOCAL_TRAIN_PATH}")
print(f"Saved testing data to: {LOCAL_TEST_PATH}")

# 6. Upload the split datasets to Hugging Face
print(f"Uploading {LOCAL_TRAIN_PATH} to {HF_DATASET_REPO}/tourism_train.csv...")
upload_file(
    path_or_fileobj=LOCAL_TRAIN_PATH,
    path_in_repo='tourism_train.csv',
    repo_id=HF_DATASET_REPO,
    repo_type='dataset',
    token=HF_TOKEN
)
print(f"Uploaded {LOCAL_TRAIN_PATH}")

print(f"Uploading {LOCAL_TEST_PATH} to {HF_DATASET_REPO}/tourism_test.csv...")
upload_file(
    path_or_fileobj=LOCAL_TEST_PATH,
    path_in_repo='tourism_test.csv',
    repo_id=HF_DATASET_REPO,
    repo_type='dataset',
    token=HF_TOKEN
)
print(f"Uploaded {LOCAL_TEST_PATH}")

print("Data preparation complete and datasets uploaded.")
