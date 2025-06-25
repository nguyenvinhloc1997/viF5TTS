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
            "ckpts/model_last.pt",  # Main model checkpoint
            "vi-fine-tuned-f5-tts.yaml",       # Configuration file
            "vocab.txt"                         # Vocabulary file
        ],
        "local_dir": "models/danhtran2mind_vi-f5-tts",
        "needs_custom_config": True,
        "config_file": "vi-fine-tuned-f5-tts.yaml"
    },
    "erax-ai/EraX-Smile-UnixSex-F5": {
        "name": "EraX Smile UnixSex F5",
        "description": "Vietnamese unisex voice F5-TTS model (supports both male and female)",
        "files": [
            "models/model_48000.safetensors",
            "models/overfit.safetensors",
            "models/config.json",
            "models/vocab.txt",
            "models/F5TTS_v1_Base.yaml"
        ],
        "local_dir": "models/erax-ai_EraX-Smile-UnixSex-F5",
        "needs_custom_config": False
    },
}

def create_model_folder_name(repo_id):
    """Create a clean folder name from repository ID"""
    return repo_id.replace("/", "_").replace("-", "_")

def create_custom_config_symlink(model_info):
    """Create symbolic link for custom config files to work with F5TTSWrapper"""
    if not model_info.get("needs_custom_config"):
        return True
    
    config_file = model_info.get("config_file")
    if not config_file:
        logging.warning("Model needs custom config but no config_file specified")
        return False
    
    # Source: actual config file in model directory
    source_config = Path(model_info["local_dir"]) / config_file
    
    # Target: symlink with "custom" in name in models root
    target_config = Path("models") / f"custom-{model_info['local_dir'].split('/')[-1]}-config.yaml"
    
    try:
        # Remove existing symlink if it exists
        if target_config.exists() or target_config.is_symlink():
            target_config.unlink()
        
        # Create relative symlink
        relative_source = os.path.relpath(source_config, target_config.parent)
        target_config.symlink_to(relative_source)
        
        logging.info(f"✓ Created custom config symlink: {target_config} -> {source_config}")
        logging.info(f"  Use this in F5TTSWrapper: model_name='{target_config}'")
        return True
        
    except Exception as e:
        logging.error(f"✗ Failed to create custom config symlink: {e}")
        return False

def setup_model_files(model_info):
    """Setup model files after download (create symlinks, etc.)"""
    logging.info("Setting up model files...")
    
    # Create custom config symlink if needed
    if model_info.get("needs_custom_config"):
        if not create_custom_config_symlink(model_info):
            logging.warning("Custom config symlink creation failed, model may not work properly")
    
    # Copy commonly used files to models root for easy access
    model_dir = Path(model_info["local_dir"])
    models_root = Path("models")
    
    # Check for vocab.txt and copy to models root if it doesn't exist
    vocab_candidates = [
        model_dir / "vocab.txt",
        model_dir / "models" / "vocab.txt"  # For erax models
    ]
    
    root_vocab = models_root / "vocab.txt"
    if not root_vocab.exists():
        for vocab_path in vocab_candidates:
            if vocab_path.exists():
                import shutil
                shutil.copy2(vocab_path, root_vocab)
                logging.info(f"✓ Copied vocab.txt to models root: {vocab_path} -> {root_vocab}")
                break
    
    # Check for model files and copy/symlink popular ones to models root
    model_candidates = [
        (model_dir / "models" / "overfit.safetensors", models_root / "overfit.safetensors"),
        (model_dir / "models" / "model_48000.safetensors", models_root / "model_48000.safetensors"),
    ]
    
    for source, target in model_candidates:
        if source.exists() and not target.exists():
            try:
                # Create relative symlink
                relative_source = os.path.relpath(source, target.parent)
                target.symlink_to(relative_source)
                logging.info(f"✓ Created model symlink: {target} -> {source}")
            except Exception as e:
                logging.debug(f"Could not create symlink for {source}: {e}")

def list_available_models():
    """List all available models"""
    logging.info("Available models:")
    for i, (repo_id, info) in enumerate(AVAILABLE_MODELS.items(), 1):
        logging.info(f"{i}. {info['name']}")
        logging.info(f"   Repository: {repo_id}")
        logging.info(f"   Description: {info['description']}")
        logging.info(f"   Local folder: {info['local_dir']}")
        if info.get("needs_custom_config"):
            logging.info(f"   ⚙️  Uses custom config: {info.get('config_file')}")
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
    if repo_id in AVAILABLE_MODELS:
        model_info = AVAILABLE_MODELS[repo_id]
    else:
        # Handle unknown models by creating a default configuration
        logging.warning(f"Model {repo_id} not in pre-configured list, using auto-detection...")
        folder_name = create_model_folder_name(repo_id)
        model_info = {
            "name": f"Custom model: {repo_id}",
            "description": "Auto-detected model",
            "files": custom_files or [],
            "local_dir": f"models/{folder_name}",
            "needs_custom_config": False
        }
    
    logging.info(f"Downloading model: {model_info['name']}")
    logging.info(f"Repository: {repo_id}")
    logging.info(f"Local directory: {model_info['local_dir']}")
    
    # Use custom files if provided, otherwise use default files
    files_to_download = custom_files if custom_files else model_info['files']
    
    # Check if we can list files in the repo to verify they exist
    try:
        repo_files = list_repo_files(repo_id)
        logging.info(f"Repository contains {len(repo_files)} files")
        
        # If no files specified, download all files (excluding hidden files and directories)
        if not files_to_download:
            files_to_download = [f for f in repo_files if not f.startswith('.') and '/' not in f]
            logging.info("No specific files configured, will download all root-level files")
        else:
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
    
    # Setup model files (symlinks, etc.)
    setup_model_files(model_info)
    
    logging.info(f"🎉 Model download completed!")
    logging.info(f"📁 Files saved to: {model_info['local_dir']}/")
    
    if model_info.get("needs_custom_config"):
        custom_config_link = Path("models") / f"custom-{model_info['local_dir'].split('/')[-1]}-config.yaml"
        logging.info(f"🔗 Custom config available at: {custom_config_link}")
        logging.info(f"   Use this in F5TTSWrapper: model_name='{custom_config_link}'")
    
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