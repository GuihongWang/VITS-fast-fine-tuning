import whisper
import os
import json
import torchaudio
import argparse
import torch
import sys

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

def process_wav_file(wav_path, save_path, model, target_sr, lang2token, processed_files, output_file, speaker):
    """Process a WAV file and append to output file if not already processed."""
    if save_path in processed_files:
        print(f"Skipping already processed file: {save_path}")
        return
    try:
        wav, sr = torchaudio.load(wav_path)
        wav = wav.mean(dim=0).unsqueeze(0)  # Convert to mono
        if sr != target_sr:
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)
        duration = wav.shape[1] / target_sr
        if duration > 20.0:
            print(f"{wav_path} too long ({duration:.2f}s > 20.0s), ignoring")
            return
        torchaudio.save(save_path, wav, target_sr, channels_first=True)
        lang, text = transcribe_one(save_path, model)
        if lang is None or text is None:
            return
        if lang not in lang2token:
            print(f"{lang} not supported for {wav_path}, ignoring")
            return
        annotated_text = f"{lang2token[lang]}{text}{lang2token[lang]}"  # No extra newline
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"{save_path}|{speaker}|{annotated_text}\n")  # Newline only at end of line
        print(f"Processed: {save_path}")
    except Exception as e:
        print(f"Error processing {wav_path}: {e}", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", default="CJE", help="Language set: CJE (Chinese, Japanese, English), CJ (Chinese, Japanese), C (Chinese)")
    parser.add_argument("--whisper_size", default="medium", help="Whisper model size")
    parser.add_argument("--input_dir", default="./custom_character_voice/", help="Input audio directory")
    parser.add_argument("--output_file", default="short_character_anno.txt", help="Output transcription file")
    parser.add_argument("--config_file", default="./configs/finetune_speaker.json", help="Path to config file")
    args = parser.parse_args()

    # Set lang2token based on --languages, matching original script
    if args.languages == "CJE":
        lang2token = {'zh': "[ZH]", 'ja': "[JA]", 'en': "[EN]"}
    elif args.languages == "CJ":
        lang2token = {'zh': "[ZH]", 'ja': "[JA]"}
    elif args.languages == "C":
        lang2token = {'zh': "[ZH]"}
    else:
        print(f"Invalid language set: {args.languages}", file=sys.stderr)
        sys.exit(1)

    assert torch.cuda.is_available(), "Please enable GPU to run Whisper!"
    model = whisper.load_model(args.whisper_size)

    # Load target sampling rate from config if exists
    target_sr = 16000  # Default sampling rate
    if os.path.exists(args.config_file):
        try:
            with open(args.config_file, 'r', encoding='utf-8') as f:
                hps = json.load(f)
            target_sr = hps.get('data', {}).get('sampling_rate', target_sr)
        except Exception as e:
            print(f"Error loading config file {args.config_file}: {e}, using default sampling rate {target_sr}", file=sys.stderr)

    # Load already processed files for breakpoint resumption
    processed_files = load_processed_files(args.output_file)

    speaker_names = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
    total_files = sum([len([f for f in files if f.endswith('.wav') and not f.startswith('processed_')]) 
                       for r, d, files in os.walk(args.input_dir)])

    processed_count = 0
    for speaker in speaker_names:
        speaker_dir = os.path.join(args.input_dir, speaker)
        wav_files = [f for f in os.listdir(speaker_dir) if f.endswith('.wav') and not f.startswith('processed_')]
        for i, wavfile in enumerate(wav_files):
            wav_path = os.path.join(speaker_dir, wavfile)
            save_path = os.path.join(speaker_dir, f"processed_{i}.wav")
            process_wav_file(wav_path, save_path, model, target_sr, lang2token, processed_files, args.output_file, speaker)
            processed_count += 1
            print(f"Processed: {processed_count}/{total_files}")

    if processed_count == 0:
        print("Warning: no short audios found. This is expected if only long audios or unsupported languages were provided.")
