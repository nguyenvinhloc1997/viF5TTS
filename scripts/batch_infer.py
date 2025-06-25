# Ubuntu: sudo apt install ffmpeg
# Windows please refer to https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/

import os
import json
import logging
from datetime import datetime
os.environ["CUDA_VISIBLE_DEVICES"] =  "0" # Tell it which GPU to use (or ignore if you're CPU-bound and patient!)

from vinorm import TTSnorm # Gotta normalize that Vietnamese text first
from f5_tts.infer.f5tts_wrapper import F5TTSWrapper # Our handy wrapper class

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# --- Configuration ---
# Path to the model checkpoint
eraX_ckpt_path = "models/overfit.safetensors"

# Path to the vocab file
vocab_file = "models/vocab.txt"

# Base output directory
base_output_dir = "outputs"

# List of sentences to generate for QA
sentences_to_generate = [
    "dạ",
    "vâng",
    "Vâng em cảm ơn anh. Khi nào có nhu cầu thì anh liên hệ lại cho bên em nha.",
    "Bên em đang giới thiệu đợt đầu một đại đô thị nghỉ dưỡng phía Tây Hà Nội, chính sách cực tốt cho cả khách mua ở và đầu tư. Mình hiện quan tâm mua để ở hay đầu tư sinh lời ạ",
    "Em xin cảm ơn ạ, bên em sẽ liên hệ anh qua Zalo nha. Em chào anh.",
    "Em xin phép kết bạn Zalo để các bạn chuyên viên phụ trách dự án sẽ liên hệ tư vấn cho anh kĩ hơn nha. Anh có dùng Zalo số này không ạ.",
    "Căn hộ mới ở Green Park Diamond Hill Central Plaza đang mở bán, em có thể tư vấn cho anh ạ."
]

# Load voice configurations
def load_all_voice_configs():
    with open("data/voice_configs.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    return config["voices"]

# Load all available voices
voice_configs = load_all_voice_configs()

logging.info("=== Batch Voice Generation for QA ===")
logging.info(f"Found {len(voice_configs)} voices to process")
logging.info(f"Will generate {len(sentences_to_generate)} sentences per voice")
logging.info(f"Total audio files to generate: {len(voice_configs) * len(sentences_to_generate)}")

# Create timestamped output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_root = os.path.join(base_output_dir, f"batch_{timestamp}")
os.makedirs(output_root, exist_ok=True)

# Initialize TTS engine once
logging.info("Initializing the TTS engine... (This might take a moment)")
tts = F5TTSWrapper(
    vocoder_name="vocos",
    ckpt_path=eraX_ckpt_path,
    vocab_file=vocab_file,
    use_ema=False,
)

# Process each voice
for voice_idx, voice_config in enumerate(voice_configs, 1):
    voice_id = voice_config["voice_id"]
    
    logging.info(f"[{voice_idx}/{len(voice_configs)}] Processing voice: {voice_id}")
    logging.info(f"  - Gender: {voice_config['gender']}")
    logging.info(f"  - Accent: {voice_config['accent']}")
    logging.info(f"  - Speed: {voice_config['default_speed']}")
    
    # Create voice-specific output directory
    voice_output_dir = os.path.join(output_root, voice_id)
    os.makedirs(voice_output_dir, exist_ok=True)
    
    # Set up voice-specific paths
    ref_audio_path = f"voice_samples/{voice_config['filename']}"
    ref_text = voice_config["reference_text"]
    speed = voice_config.get("default_speed", 1.0)
    
    # Normalize reference text
    ref_text_norm = TTSnorm(ref_text)
    
    # Preprocess reference audio for this voice
    logging.info(f"  - Processing reference voice: {voice_config['filename']}")
    tts.preprocess_reference(
        ref_audio_path=ref_audio_path,
        ref_text=ref_text_norm,
        clip_short=True
    )
    logging.info(f"  - Reference audio duration: {tts.get_current_audio_length():.2f}s")
    
    # Generate audio for each sentence
    for sent_idx, sentence in enumerate(sentences_to_generate, 1):
        logging.debug(f"    Generating sentence {sent_idx}/{len(sentences_to_generate)}...")
        
        # Normalize the sentence
        text_norm = TTSnorm(sentence)
        
        # Create output filename
        output_filename = f"sentence_{sent_idx:02d}.wav"
        output_path = os.path.join(voice_output_dir, output_filename)
        
        # Generate the audio
        tts.generate(
            text=text_norm,
            output_path=output_path,
            nfe_step=32,
            cfg_strength=3.0,
            speed=speed,
            cross_fade_duration=0.12,
        )
    
    logging.info(f"  ✓ Completed {voice_id} - {len(sentences_to_generate)} files generated")

logging.info("🎉 Batch generation completed!")
logging.info(f"📁 Output directory: {output_root}")
logging.info(f"🔊 Total audio files generated: {len(voice_configs) * len(sentences_to_generate)}")
logging.info("Ready for QA! Check each voice folder to compare audio quality.") 