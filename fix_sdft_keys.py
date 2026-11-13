"""Rename safetensors keys: insert .linear. before .weight for 232 keys."""
import json
import os
from safetensors import safe_open
from safetensors.torch import save_file
import torch

MODEL_DIR = "/models/gemma-4-E4B-it-SDFT_Heretic-runpod-vllm"
INDEX_PATH = os.path.join(MODEL_DIR, "model.safetensors.index.json")

# Load index
with open(INDEX_PATH) as f:
    idx = json.load(f)

# Build rename map: keys that need .linear. inserted
# Pattern: X.weight -> X.linear.weight for specific module types
# We identify them by matching against the original model's index
ORIG_INDEX = "/models/gemma-4-E4B-it-SDFT_Heretic_RP/model.safetensors.index.json"
with open(ORIG_INDEX) as f:
    orig_idx = json.load(f)

orig_keys = set(orig_idx['weight_map'].keys())
new_keys = set(idx['weight_map'].keys())

# Find keys in original with .linear.weight that exist in new as .weight
rename_map = {}
for orig_key in orig_keys:
    if '.linear.weight' not in orig_key:
        continue
    without_linear = orig_key.replace('.linear.weight', '.weight')
    if without_linear in new_keys:
        rename_map[without_linear] = orig_key

print(f"Keys to rename: {len(rename_map)}")

# Group by shard
shard_renames = {}
for new_key, orig_key in rename_map.items():
    shard = idx['weight_map'][new_key]
    if shard not in shard_renames:
        shard_renames[shard] = {}
    shard_renames[shard][new_key] = orig_key

for shard, renames in sorted(shard_renames.items()):
    print(f"  {shard}: {len(renames)} keys")

# Process each affected shard
for shard_name, renames in shard_renames.items():
    shard_path = os.path.join(MODEL_DIR, shard_name)
    print(f"\nProcessing {shard_name}...")
    
    # Load all tensors from this shard
    tensors = {}
    with safe_open(shard_path, framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    
    print(f"  Loaded {len(tensors)} tensors")
    
    # Rename keys
    renamed = {}
    renamed_count = 0
    for key, tensor in tensors.items():
        if key in renames:
            new_key = renames[key]
            renamed[new_key] = tensor
            renamed_count += 1
        else:
            renamed[key] = tensor
    
    print(f"  Renamed {renamed_count} keys")
    
    # Save back (backup first)
    backup_path = shard_path + ".bak"
    if not os.path.exists(backup_path):
        os.rename(shard_path, backup_path)
        print(f"  Backed up to {backup_path}")
    
    save_file(renamed, shard_path)
    print(f"  Saved {shard_path}")
    
    # Update index
    for old_key, new_key in renames.items():
        idx['weight_map'][new_key] = idx['weight_map'].pop(old_key)

# Save updated index
with open(INDEX_PATH, 'w') as f:
    json.dump(idx, f, indent=2)
print(f"\nUpdated index: {INDEX_PATH}")
print("Done!")
