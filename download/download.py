from datasets import load_dataset
ds = load_dataset("didsr/msynth", "device_data", trust_remote_code=True)

# For device data for all breast density, mass redius, mass density, and relative dose, change configuration to
# 'segmentation_mask' and 'metadata' to load the segmentation masks and bound information
print(ds_data["device_data"])


