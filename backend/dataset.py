import kagglehub

# Download latest version
path = kagglehub.dataset_download("hamnakaleemds/global-startup-success-dataset")

print("Path to dataset files:", path)
