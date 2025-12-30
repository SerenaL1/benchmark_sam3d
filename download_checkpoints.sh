TAG=hf
hf download \
  --repo-type model \
  --local-dir checkpoints/ \
  --max-workers 1 \
  facebook/sam-3d-objects
# mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}
# rm -rf checkpoints/${TAG}-download