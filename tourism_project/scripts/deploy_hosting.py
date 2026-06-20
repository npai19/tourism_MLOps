from huggingface_hub import HfApi, create_repo, upload_folder, upload_file
import os
import sys

# Retrieve environment variables
HF_TOKEN = os.environ.get('HF_TOKEN')
HF_SPACE_REPO = os.environ.get('HF_SPACE_REPO') # e.g., 'your-username/tourism_prediction_app'

if not HF_TOKEN or not HF_SPACE_REPO:
    print("Error: HF_TOKEN and HF_SPACE_REPO environment variables must be set.")
    sys.exit(1)

# Define paths
LOCAL_DEPLOYMENT_DIR = 'tourism_project/deployment'
LOCAL_MODEL_PATH = 'tourism_project/model_building/best_decision_tree_model.joblib'

# Create a new Space repository on Hugging Face Hub
# This will create a public Space by default
create_repo(repo_id=HF_SPACE_REPO, repo_type='space', space_sdk='docker', token=HF_TOKEN, exist_ok=True)
print(f"Created/Checked Hugging Face Space repo: {HF_SPACE_REPO}")

api = HfApi()

# Upload all files from the local deployment directory to the Hugging Face Space
print(f"Uploading deployment files from {LOCAL_DEPLOYMENT_DIR} to {HF_SPACE_REPO}...")
upload_folder(
    folder_path=LOCAL_DEPLOYMENT_DIR,
    repo_id=HF_SPACE_REPO,
    repo_type='space',
    token=HF_TOKEN,
    commit_message='Update Streamlit app, Dockerfile, and requirements.txt'
)
print(f"Uploaded deployment files.")

# Upload the model file to the Hugging Face Space
print(f"Uploading {LOCAL_MODEL_PATH} to {HF_SPACE_REPO}/best_decision_tree_model.joblib...")
api.upload_file(
    path_or_fileobj=LOCAL_MODEL_PATH,
    path_in_repo='best_decision_tree_model.joblib',
    repo_id=HF_SPACE_REPO,
    repo_type='space',
    token=HF_TOKEN,
    commit_message='Update best trained model'
)
print(f"Uploaded {LOCAL_MODEL_PATH}")

print(f"Hugging Face Space deployment complete. Check your app at: https://huggingface.co/spaces/{HF_SPACE_REPO.split('/')[-2]}/{HF_SPACE_REPO.split('/')[-1]}")
