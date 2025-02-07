from whisper import load_model
import os
import csv
import numpy as np
import librosa
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def get_whisper_folder():
    while True:
        folder_path = "/home/shaun/ai/Audio To Transcribe"
        if not folder_path:  # Default to the current directory
            folder_path = os.getcwd()
        if os.path.isdir(folder_path):
            return os.path.abspath(folder_path)
        print("Invalid folder path. Please try again.")

def is_valid_file(filename):
    valid_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.mp4']
    return any(filename.lower().endswith(ext) for ext in valid_extensions)

def detect_pauses(audio_path, min_pause_duration=0.5):
    y, sr = librosa.load(audio_path)
   
    # Detect pauses using energy levels
    energy = librosa.feature.rms(y=y)[0]
    silence_threshold = np.mean(energy) * 0.1
    silent_frames = energy < silence_threshold
   
    pauses = []
    pause_start = None
    for i, is_silent in enumerate(silent_frames):
        time = librosa.frames_to_time(i, sr=sr)
        if is_silent and pause_start is None:
            pause_start = time
        elif not is_silent and pause_start is not None:
            pause_duration = time - pause_start
            if pause_duration >= min_pause_duration:
                pauses.append((pause_start, time))
            pause_start = None
   
    return pauses

def format_timestamp(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def transcribe_media(media_filename, model, model_name, folder_path):
    try:
        input_name, _ = os.path.splitext(media_filename)
        csv_filename = f"{input_name}_transcription.csv"
        txt_filename = f"{input_name}_transcription.txt"

        # Define paths for output files
        csv_path = os.path.join(folder_path, csv_filename)
        txt_path = os.path.join(folder_path, txt_filename)

        media_path = os.path.join(folder_path, media_filename)

        print(f"Transcribing: {media_filename}")

        # Perform transcription
        result = model.transcribe(media_path)

        # Write CSV with timestamps and pauses
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["Start Time", "End Time", "Text"])
            
            for segment in result["segments"]:
                csvwriter.writerow([
                    format_timestamp(segment['start']),
                    format_timestamp(segment['end']),
                    segment['text']
                ])

        # Write text file
        with open(txt_path, 'w', encoding='utf-8') as txtfile:
            for segment in result["segments"]:
                txtfile.write(f"{segment['text']}\n")

        print(f"Transcription completed and saved to:")
        print(f"CSV: {csv_path}")
        print(f"Text: {txt_path}")

    except Exception as e:
        print(f"An error occurred while processing {media_filename}: {e}")


def get_model_choice():
    while True:
        choice = input("Choose Whisper model (base/small/medium/large): ").lower().strip()
        if choice in ['base', 'small', 'medium', 'large']:
            return choice
        print("Invalid choice. Please enter 'base', 'small', 'medium', or 'large'.")

def process_all_media_files():
    whisper_folder = get_whisper_folder()
    os.chdir(whisper_folder)
   
    model_name = get_model_choice()
   
    print(f"Loading Whisper {model_name} model...")
    model = load_model(model_name)
   
    media_files = [f for f in os.listdir(whisper_folder) if is_valid_file(f)]
   
    if not media_files:
        print("No valid audio or video files found in the specified folder.")
        return
   
    print(f"Found {len(media_files)} valid file(s). Starting transcription process...")
   
    for media_file in media_files:
        transcribe_media(media_file, model, model_name, whisper_folder)
   
    print("All transcriptions completed.")

if __name__ == "__main__":
    process_all_media_files()
