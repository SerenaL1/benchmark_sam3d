from huggingface_hub import snapshot_download

# Download the specific 'imagesTr_ct' folder
# allow_patterns ensures only files inside that directory are downloaded
local_path = snapshot_download(
    repo_id="blueyo0/SA-Med3D-140K",
    repo_type="dataset",
    allow_patterns="**/imagesTr_ct/**",  # Matches the folder and all its contents
    local_dir="/PHShome/yl535/project/python/datasets/SA-Med3D-140K_imagesTr_ct", # Where to save it locally
    resume_download=True
)

print(f"Downloaded to: {local_path}")