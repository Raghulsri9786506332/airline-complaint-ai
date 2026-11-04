"""
scripts/upload_models_to_hf.py — Upload model artifacts to Hugging Face Hub.

Run once after creating HF repo:
    python scripts/upload_models_to_hf.py --repo-id YOUR_USERNAME/airline-models --token HF_TOKEN

Files uploaded:
  - tfidf_vectorizer.pkl
  - severity_Logistic_Regression_tfidf.pkl
  - risk_Logistic_Regression_tfidf_struct.pkl
  - struct_scaler.pkl
"""
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parents[1]

MODEL_FILES = [
    ("artifacts/vectorizers/tfidf_vectorizer.pkl", "tfidf_vectorizer.pkl"),
    ("models/ml/severity_Logistic_Regression_tfidf.pkl", "severity_Logistic_Regression_tfidf.pkl"),
    ("models/ml/risk_Logistic_Regression_tfidf_struct.pkl", "risk_Logistic_Regression_tfidf_struct.pkl"),
    ("artifacts/vectorizers/struct_scaler.pkl", "struct_scaler.pkl"),
]

def main():
    parser = argparse.ArgumentParser(description="Upload model artifacts to HF Hub")
    parser.add_argument("--repo-id", required=True, help="HF repo ID (e.g., username/airline-models)")
    parser.add_argument("--token", required=True, help="HF token with write access")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    args = parser.parse_args()

    api = HfApi(token=args.token)

    # Create repo if not exists
    try:
        create_repo(args.repo_id, token=args.token, private=args.private, exist_ok=True)
        print(f"✅ Repo ready: {args.repo_id}")
    except Exception as e:
        print(f"⚠️ Repo creation: {e}")

    # Upload each file
    for local_path, repo_path in MODEL_FILES:
        full_local = ROOT / local_path
        if not full_local.exists():
            print(f"❌ Missing: {full_local}")
            continue
        print(f"⬆️ Uploading {local_path} → {repo_path} …")
        api.upload_file(
            path_or_fileobj=str(full_local),
            path_in_repo=repo_path,
            repo_id=args.repo_id,
            token=args.token,
        )
        print(f"  ✅ Done")

    print(f"\n🎉 All files uploaded to https://huggingface.co/{args.repo_id}")

if __name__ == "__main__":
    main()