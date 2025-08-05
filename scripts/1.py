import os
import argparse
import sys
import torchaudio
import whisper
import torch
import json

lang2token = {
    'zh': "[ZH]",
    'ja': "[JA]",
    "en": "[EN]",
}

def load_processed_files(output_file):
    """Load paths of already processed files from the output file."""
    processed_files = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 1:
                    processed_files.add(parts[0])
    return processed_files

def transcribe_one(audio_path, model):
    """Transcribe a single audio file using Whisper."""
    try:
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        _, probs = model.detect_language(mel)
        lang = max(probs, key=probs.get)
        print(f"Detected language for {audio_path}: {lang}")
        options = whisper.DecodingOptions(beam_size=5)
        result = whisper.decode(model, mel, options)
        print(f"Transcription: {result.text}")
        return lang, result.text
    except Exception as e:
        print(f"Error transcribing {audio_path}: {e}", file=sys.stderr)
        return None, None

def process_wav_file(wav_path, save_path, model, target_sr):
    """Process a WAV file and return the annotated transcription."""
    try:
        wav, sr = torchaudio.load(wav_path)
        wav = wav.mean(dim=0).unsqueeze(0)  # Convert to mono
        if sr != target_sr:
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)
        duration = wav.shape[1] / target_sr
        if duration > 20.0:  # Max duration 20 seconds
            print(f"{wav_path} too long ({duration:.2f}s > 20.0s), ignoring")
            return None
        torchaudio.save(save_path, wav, target_sr, channels_first=True)
        lang, text = transcribe_one(save_path, model)
        if lang is None or text is None:
            return None
        if lang not in lang2token:
            print(f"{lang} not supported for {wav_path}, ignoring")
            return None
        annotated_text = lang2token[lang] + text + lang2token[lang] + "\n"
        return annotated_text
    except Exception as e:
        print(f"Error processing {wav_path}: {e}", file=sys.stderr)
        return None

def main(args):
    # Load Whisper model
    assert torch.cuda.is_available(), "Please enable GPU to run Whisper!"
    model = whisper.load_model(args.whisper_size)

    # Load target sampling rate from config
    with open(args.config_file, 'r', encoding='utf-8') as f:
        hps = json.load(f)
    target_sr = hps['data']['sampling_rate']

    # Load already processed files
    processed_files = load_processed_files(args.output_file)

    parent_dir = args.input_dir
    speaker_names = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]

    for speaker in speaker_names:
        speaker_dir = os.path.join(parent_dir, speaker)
        # Only process original WAV files (exclude processed ones)
        wav_files = [f for f in os.listdir(speaker_dir) 
                     if f.endswith('.wav') and not f.startswith('processed_')]

        for i, wavfile in enumerate(wav_files):
            wav_path = os.path.join(speaker_dir, wavfile)
            save_path = os.path.join(speaker_dir, f"processed_{i}.wav")

            # Skip if already processed and not forcing reprocess
            if save_path in processed_files and not args.force_process:
                print(f"Skipping already processed file: {save_path}")
                continue

            annotated_text = process_wav_file(wav_path, save_path, model, target_sr)
            if annotated_text:
                # Append to output file immediately
                with open(args.output_file, 'a', encoding='utf-8') as f:
                    f.write(f"{save_path}|{speaker}|{annotated_text}")
                print(f"Processed: {save_path}")
            else:
                print(f"Failed to process {wav_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio file processing script with breakpoint continuation")
    parser.add_argument("--input_dir", required=True, help="Input audio directory")
    parser.add_argument("--output_file", required=True, help="Output transcription file")
    parser.add_argument("--config_file", required=True, help="Path to config file")
    parser.add_argument("--whisper_size", default="medium", help="Whisper model size")
    parser.add_argument("--force_process", action="store_true", help="Force reprocess all files")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist.", file=sys.stderr)
        sys.exit(1)

    main(args)
