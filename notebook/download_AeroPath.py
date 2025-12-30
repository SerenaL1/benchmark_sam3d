from huggingface_hub import snapshot_download

# 1. Define repository and destination
repo_id = "andreped/AeroPath"
destination_dir = "/PHShome/yl535/project/python/datasets/AeroPath"

# 2. Download the snapshot
# repo_type="dataset" is crucial because it defaults to "model" otherwise
local_path = snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=destination_dir,
    local_dir_use_symlinks=False  # Set to False to get actual files, not symlinks
)

print(f"Files successfully downloaded to: {local_path}")