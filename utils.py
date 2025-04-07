import os
import requests
from tqdm import tqdm

def download_model(url, save_path, expected_size=None):
    if os.path.exists(save_path):
        if expected_size is None or os.path.getsize(save_path) >= expected_size:
            print(f"{save_path} already exists and looks good. Skipping download.")
            return
        else:
            print(f"{save_path} exists but seems incomplete. Re-downloading...")

    print(f"Downloading model from {url}...")
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))

    with open(save_path, 'wb') as file, tqdm(
        desc=save_path,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
