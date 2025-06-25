#!/usr/bin/env python3
"""
Model Download Script for Vietnamese F5-TTS
Downloads models and associated files from Hugging Face repositories
"""

import os
import logging
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Available models configuration
AVAILABLE_MODELS = {
    "danhtran2mind/vi-f5-tts": {
        "name": "Vietnamese F5-TTS by danhtran2mind",
        "description": "Vietnamese text-to-speech model fine-tuned on F5-TTS",
        "files": [
            "ckpts/model_1200000.safetensors",  # Main model checkpoint
            "vi-fine-tuned-f5-tts.yaml",       # Configuration file
            "vocab.txt"                         # Vocabulary file
        ],
        "local_dir": "models"
    },
    "erax-ai/EraX-Smile-Female-F5-V1.0": {
        "name": "EraX Smile Female F5 V1.0",
        "description": "Vietnamese female voice F5-TTS model",
        "files": [
            "model.pth",
            "vocab.txt"
        ],
        "local_dir": "models"
    }
}

def list_available_models():
    """List all available models"""
    logging.info("Available models:")
    for i, (repo_id, info) in enumerate(AVAILABLE_MODELS.items(), 1):
        logging.info(f"{i}. {info['name']}")
        logging.info(f"   Repository: {repo_id}")
        logging.info(f"   Description: {info['description']}")
        logging.info("")

def download_file_with_progress(repo_id, filename, local_dir):
    """Download a single file with progress tracking"""
    try:
        logging.info(f"Downloading {filename}...")
        
        # Create local directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)
        
        # Download the file
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        
        # Get file size for logging
        file_size = os.path.getsize(downloaded_path)
        file_size_mb = file_size / (1024 * 1024)
        
        logging.info(f"✓ Downloaded {filename} ({file_size_mb:.1f} MB)")
        return downloaded_path
        
    except Exception as e:
        logging.error(f"✗ Failed to download {filename}: {str(e)}")
        return None

def download_model(repo_id, custom_files=None):
    """Download a model and its associated files"""
    if repo_id not in AVAILABLE_MODELS:
        logging.error(f"Model {repo_id} not found in available models")
        list_available_models()
        return False
    
    model_info = AVAILABLE_MODELS[repo_id]
    
    logging.info(f"Downloading model: {model_info['name']}")
    logging.info(f"Repository: {repo_id}")
    logging.info(f"Local directory: {model_info['local_dir']}")
    
    # Use custom files if provided, otherwise use default files
    files_to_download = custom_files if custom_files else model_info['files']
    
    # Check if we can list files in the repo to verify they exist
    try:
        repo_files = list_repo_files(repo_id)
        logging.info(f"Repository contains {len(repo_files)} files")
        
        # Filter files that actually exist in the repo
        available_files = []
        for file in files_to_download:
            if file in repo_files:
                available_files.append(file)
            else:
                logging.warning(f"File {file} not found in repository, skipping...")
        
        files_to_download = available_files
        
    except Exception as e:
        logging.warning(f"Could not list repository files: {e}")
        logging.info("Proceeding with configured file list...")
    
    if not files_to_download:
        logging.error("No files to download!")
        return False
    
    logging.info(f"Will download {len(files_to_download)} files:")
    for file in files_to_download:
        logging.info(f"  - {file}")
    
    # Download each file
    success_count = 0
    failed_files = []
    
    for filename in files_to_download:
        result = download_file_with_progress(repo_id, filename, model_info['local_dir'])
        if result:
            success_count += 1
        else:
            failed_files.append(filename)
    
    # Summary
    logging.info(f"\nDownload Summary:")
    logging.info(f"✓ Successfully downloaded: {success_count}/{len(files_to_download)} files")
    
    if failed_files:
        logging.error(f"✗ Failed to download: {failed_files}")
        return False
    
    logging.info(f"🎉 Model download completed!")
    logging.info(f"📁 Files saved to: {model_info['local_dir']}/")
    
    return True

def interactive_download():
    """Interactive model selection and download"""
    print("🤖 Vietnamese F5-TTS Model Downloader")
    print("=" * 50)
    
    list_available_models()
    
    try:
        choice = input("Select a model to download (enter number): ").strip()
        model_index = int(choice) - 1
        
        if 0 <= model_index < len(AVAILABLE_MODELS):
            repo_id = list(AVAILABLE_MODELS.keys())[model_index]
            return download_model(repo_id)
        else:
            print("Invalid selection")
            return False
            
    except (ValueError, KeyboardInterrupt):
        print("\nDownload cancelled")
        return False

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Vietnamese F5-TTS models from Hugging Face")
    parser.add_argument("--model", "-m", help="Model repository ID (e.g., danhtran2mind/vi-f5-tts)")
    parser.add_argument("--list", "-l", action="store_true", help="List available models")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive model selection")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
    elif args.interactive:
        interactive_download()
    elif args.model:
        download_model(args.model)
    else:
        # Default to interactive mode if no arguments provided
        interactive_download()

if __name__ == "__main__":
    main() 