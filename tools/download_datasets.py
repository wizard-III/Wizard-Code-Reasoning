#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    :   2025/06/17 19:18:58
@Author  :   wangjiakang
@File    :   __init__.py
'''

import os
import shutil
from huggingface_hub import hf_hub_download

def download_from_hf(repo_id, filename, save_path):
    """Download a file from Hugging Face Hub and save to specified path
    
    Args:
        repo_id: Repository ID on Hugging Face Hub
        filename: Relative path of file in the repository
        save_path: Local path to save the downloaded file
    """
    # Ensure target directory exists
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"Created directory: {save_dir}")
    
    # Download file
    file_path = hf_hub_download(
        repo_id=repo_id,
        repo_type='dataset',
        filename=filename
    )
    
    # Copy to destination
    shutil.copyfile(file_path, save_path)
    print(f"File saved to: {save_path}")

repo_id = "wizardII/ArcherCodeR-Dataset"

# Download training data
download_from_hf(
    repo_id,
    "train/archercoder-1.5b-train.json",
    "./data/train/archercoder-1.5b-train.json"
)

# Download test data
download_from_hf(
    repo_id,
    "test/livecodebench_v5.json",
    "./data/test/livecodebench_v5.json"
)

