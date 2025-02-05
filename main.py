from whisper import load_model
import os
import csv
import numpy as np
import librosa
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def get_whisper_folder():
    return os.path.abspath(os.path.dirname(__file__))

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

def transcribe_media(media_filename, model, model_name):
    whisper_folder = get_whisper_folder()
    media_path = os.path.join(whisper_folder, media_filename)

    try:
        print(f"Transcribing: {media_filename}")
       
        if not os.path.exists(media_path):
            raise FileNotFoundError(f"The file {media_path} does not exist.")

        print("Starting transcription for file:", media_filename)
       
        print("Detecting pauses...")
        pauses = detect_pauses(media_path)

        print("Transcribing media...")
        result = model.transcribe(media_path)

        input_name, _ = os.path.splitext(media_filename)
        csv_filename = f"{input_name}_transcription_{model_name}.csv"
        txt_filename = f"{input_name}_transcription_{model_name}.txt"
        csv_path = os.path.join(whisper_folder, csv_filename)
        txt_path = os.path.join(whisper_folder, txt_filename)

        # Write CSV with timestamps and pauses
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["Start Time", "End Time", "Text"])
           
            for segment in result["segments"]:
                # Check if this segment contains a pause
                has_pause = any(pause[0] <= segment['start'] <= pause[1] for pause in pauses)
                text = "[Pause]" if has_pause else segment['text'].strip()
               
                csvwriter.writerow([
                    f"{segment['start']:.2f}",
                    f"{segment['end']:.2f}",
                    text
                ])

        # Write clean text file with line breaks at pauses
        with open(txt_path, 'w', encoding='utf-8') as txtfile:
            current_text = []
            for segment in result["segments"]:
                has_pause = any(pause[0] <= segment['start'] <= pause[1] for pause in pauses)
                text = segment['text'].strip()
               
                if text:  # Only process non-empty segments
                    if has_pause and current_text:
                        # Write accumulated text and add line break
                        txtfile.write(' '.join(current_text) + '\n\n')
                        current_text = []
                    if not has_pause:
                        current_text.append(text)
           
            # Write any remaining text
            if current_text:
                txtfile.write(' '.join(current_text))

        print(f"Transcription completed and saved to:")
        print(f"CSV: {csv_path}")
        print(f"Text: {txt_path}")

    except Exception as e:
        print(f"An error occurred while processing {media_filename}: {str(e)}")

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
        print("No valid audio or video files found in the Whisper folder.")
        return
   
    print(f"Found {len(media_files)} valid file(s). Starting transcription process...")
   
    for media_file in media_files:
        transcribe_media(media_file, model, model_name)
   
    print("All transcriptions completed.")

if __name__ == "__main__":
    process_all_media_files()