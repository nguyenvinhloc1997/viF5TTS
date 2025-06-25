# Ubuntu: sudo apt install ffmpeg
# Windows please refer to https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/

import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] =  "0" # Tell it which GPU to use (or ignore if you're CPU-bound and patient!)

from vinorm import TTSnorm # Gotta normalize that Vietnamese text first
from f5_tts.infer.f5tts_wrapper import F5TTSWrapper # Our handy wrapper class

# === MODEL SELECTION ===
# Choose one of: "erax_overfit", "erax_model_48000", "danhtran_vi_f5tts"
SELECTED_MODEL = "erax_overfit"  # Change this to switch models

# Model configurations (simplified)
MODEL_CONFIGS = {
    "erax_overfit": {
        "name": "EraX Overfit Model",
        "model_name": "F5TTS_v1_Base",  # Standard config
        "ckpt_path": "models/overfit.safetensors",
        "vocab_file": "models/vocab.txt",
        "use_ema": False
    },
    "erax_model_48000": {
        "name": "EraX Model 48000",
        "model_name": "F5TTS_v1_Base",  # Standard config
        "ckpt_path": "models/model_48000.safetensors", 
        "vocab_file": "models/vocab.txt",
        "use_ema": False
    },
    "danhtran_vi_f5tts": {
        "name": "DanhTran Vietnamese F5-TTS",
        "model_name": "models/custom-danhtran2mind_vi-f5-tts-config.yaml",  # Custom config with "custom" in path
        "ckpt_path": "models/danhtran2mind_vi-f5-tts/ckpts/model_last.pt",
        "vocab_file": "models/danhtran2mind_vi-f5-tts/vocab.txt",
        "use_ema": True
    }
}

# Get selected model config
if SELECTED_MODEL not in MODEL_CONFIGS:
    raise ValueError(f"Unknown model: {SELECTED_MODEL}. Available: {list(MODEL_CONFIGS.keys())}")

model_config = MODEL_CONFIGS[SELECTED_MODEL]
print(f"Selected model: {model_config['name']}")

# --- Voice Configuration ---
# Specify which voice to use from data/voice_configs.json
voice_id = "nu_mien_nam_1"  # Available: huong_giang_4, nu_mien_bac_1, nam_mien_bac_1, nu_mien_nam_1, nam_mien_nam_1

# Load voice configuration
def load_voice_config(voice_id):
    with open("data/voice_configs.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    
    for voice in config["voices"]:
        if voice["voice_id"] == voice_id:
            return voice
    
    raise ValueError(f"Voice ID '{voice_id}' not found in voice_configs.json")

# Load the selected voice configuration
voice_config = load_voice_config(voice_id)

# --- Config ---
# Path to the voice you want to clone (from voice config)
ref_audio_path = f"voice_samples/{voice_config['filename']}"

# Where to save the generated sound
output_dir = "outputs"

# Speed setting (from config, can be overridden)
speed = 1

# --- Texts ---
# Text matching the reference audio (from voice config)
ref_text = voice_config["reference_text"]

# The text you want the cloned voice to speak
text_to_generate = "Vâng em cảm ơn anh. Khi nào có nhu cầu thì anh liên hệ lại cho bên em nha."

# --- Let's Go! ---
print(f"Using voice: {voice_config['voice_id']} ({voice_config['gender']}, {voice_config['accent']} accent)")
print(f"Reference audio: {ref_audio_path}")
print(f"Speed setting: {speed}")
print("Initializing the TTS engine... (Might take a sec)")

# Initialize with selected model config
tts = F5TTSWrapper(
    model_name=model_config["model_name"],
    ckpt_path=model_config["ckpt_path"],
    vocab_file=model_config["vocab_file"],
    vocoder_name="vocos",
    use_ema=model_config["use_ema"],
)

# Normalize the reference text (makes it easier for the model)
ref_text_norm = TTSnorm(ref_text)

# Prepare the output folder
os.makedirs(output_dir, exist_ok=True)

print("Processing the reference voice...")
# Feed the model the reference voice ONCE
# Provide ref_text for better quality, or set ref_text="" to use Whisper for auto-transcription (if installed)
tts.preprocess_reference(
    ref_audio_path=ref_audio_path,
    ref_text=ref_text_norm,
    clip_short=True # Keeps reference audio to a manageable length (~12s)
)
print(f"Reference audio duration used: {tts.get_current_audio_length():.2f} seconds")

# --- Generate New Speech ---
print("Generating new speech with the cloned voice...")

# Normalize the text we want to speak
text_norm = TTSnorm(text_to_generate)

# You can generate multiple sentences easily
# Just add more normalized strings to this list
sentences = [text_norm]

for i, sentence in enumerate(sentences):
    output_path = os.path.join(output_dir, f"{voice_config['voice_id']}_generated_speech_{i+1}.wav")

    # THE ACTUAL GENERATION HAPPENS HERE!
    tts.generate(
        text=sentence,
        output_path=output_path,
        nfe_step=20,               # Denoising steps. More = slower but potentially better? (Default: 32)
        cfg_strength=2.0,          # How strongly to stick to the reference voice style? (Default: 2.0)
        speed=speed,               # Speed from voice config (can be overridden above)
        cross_fade_duration=0.15,  # Smooths transitions if text is split into chunks (Default: 0.15)
    )
    print(f"Boom! Audio saved to: {output_path}")

print("\nAll done! Check your output folder.") 