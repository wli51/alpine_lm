from huggingface_hub import snapshot_download
from pathlib import Path



BASE_DIR = Path("/scratch/alpine/wli19@xsede.org/lms/") 
BASE_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    "unsloth/Llama-3.2-3B-Instruct",
]

for repo_id in MODELS:
    
    local_dir = BASE_DIR / repo_id.replace("/", "-")
    
    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"Skipping {repo_id}, already exists at {local_dir}")
        continue

    print(f"Downloading {repo_id} to {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,  # safer on some filesystems
        resume_download=True,
    )
    print(f"Finished {repo_id}")
