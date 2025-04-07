import os
import gdown

def download_model_from_gdrive(file_id, output_path):
    if os.path.exists(output_path):
        print(f"{output_path} already exists.")
        return
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)
