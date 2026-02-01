from huggingface_hub import snapshot_download
from pathlib import Path

REPO_ID = "ankpatil1203/legal-rag-eurlex-artifacts"
OUTPUT_DIR = Path("artifacts")

def main():
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=OUTPUT_DIR,
        local_dir_use_symlinks=False,
    )
    print(f"Artifacts downloaded to: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
