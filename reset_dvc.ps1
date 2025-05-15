# Step 1: Clean up all DVC files
Remove-Item -Recurse -Force .dvc -ErrorAction SilentlyContinue
Remove-Item -Force dvc.lock -ErrorAction SilentlyContinue
Remove-Item -Force *.dvc -ErrorAction SilentlyContinue
Remove-Item -Force **/*.dvc -ErrorAction SilentlyContinue

# Step 2: Re-initialize DVC
dvc init --no-scm

# Step 3: Add remote storage (MinIO)
dvc remote add -d minio s3://dvc
dvc remote modify minio endpointurl http://localhost:9000
dvc remote modify minio access_key_id minioadmin
dvc remote modify minio secret_access_key minioadmin

# Step 4: Add the raw data file
dvc add data/raw/symptoms2diseases.csv

# Step 5: Commit the .dvc file
git add data/raw/symptoms2diseases.csv.dvc data/raw/.gitignore .dvc/config
git commit -m "Track raw data with DVC"

# Step 6: Push to remote
dvc push