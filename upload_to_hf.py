import argparse
import os
from huggingface_hub import HfApi, create_repo

def upload_model(local_dir, repo_name, token, private=True):
    if not os.path.exists(local_dir):
        print(f"Error: Local directory {local_dir} does not exist.")
        return
        
    api = HfApi()
    
    # Extract username from token or use the default namespace
    try:
        user_info = api.whoami(token=token)
        username = user_info["name"]
    except Exception as e:
        print(f"Failed to identify HF user from token: {e}")
        print("Falling back to default repository creation...")
        username = None

    repo_id = f"{username}/{repo_name}" if username else repo_name
    print(f"Uploading {local_dir} to Hugging Face repository: {repo_id}...")

    try:
        # Create repo (no-op if it already exists)
        create_repo(repo_id=repo_id, token=token, private=private, repo_type="model", exist_ok=True)
        
        # Upload the folder content
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )
        print(f"Successfully uploaded to https://huggingface.co/{repo_id}\n")
    except Exception as e:
        print(f"Failed to upload model: {e}\n")

def main():
    parser = argparse.ArgumentParser(description="Upload fine-tuned model checkpoints to Hugging Face Hub")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Hugging Face write token (defaults to HF_TOKEN env var)")
    parser.add_argument("--public", action="store_true", help="Set repository to public (default is private)")
    args = parser.parse_args()

    # Define the models we want to upload (Targeting only the final folders)
    # Using expanding environment variables to support multiple user profiles on cluster
    user = os.environ.get("USER", "nakshatrak_iitp")
    models_to_upload = [
        {
            "local_dir": f"/scratch/{user}/checkpoints/whisper-medium-banking-configC/final",
            "repo_name": "whisper-medium-banking-configC"
        },
        {
            "local_dir": f"/scratch/{user}/checkpoints/whisper-medium-banking-configB/checkpoint-1000",  # Early stopped at step 1000
            "repo_name": "whisper-medium-banking-configB"
        },
        {
            "local_dir": f"/scratch/{user}/checkpoints/indicwav2vec-banking-configB/checkpoint-500",  # Checkpoint B final is checkpoint-500 according to config.yaml
            "repo_name": "indicwav2vec-banking-configB"
        },
        {
            "local_dir": f"/scratch/{user}/checkpoints/indicwav2vec-banking-configC/final",
            "repo_name": "indicwav2vec-banking-configC"
        }
    ]

    for model in models_to_upload:
        if os.path.exists(model["local_dir"]):
            upload_model(
                local_dir=model["local_dir"],
                repo_name=model["repo_name"],
                token=args.token,
                private=not args.public
            )
        else:
            print(f"Skipping {model['repo_name']} (Directory not found: {model['local_dir']})")

if __name__ == "__main__":
    main()
