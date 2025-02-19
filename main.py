from whisper import load_model
from langchain_ollama import OllamaLLM
import os
import csv
import numpy as np
import librosa
import warnings
import sys

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

def transcribe_media(whisper_folder, media_filename, model, model_name):
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

        # make summary
        print("Generating summary file...")
        file = open(txt_path, "r")
        text_to_summarize = file.read()
        llm = OllamaLLM(model="llama3.2")
        response = llm.invoke("Can you summarize this audio transcript, making sure to call out key points with time stamps?:" + text_to_summarize)
        print(response)

    except Exception as e:
        print(f"An error occurred while processing {media_filename}: {str(e)}")

def get_model_choice():
    while True:
        choice = input("Choose Whisper model (base/small/medium/large): ").lower().strip()
        if choice in ['base', 'small', 'medium', 'large']:
            return choice
        print("Invalid choice. Please enter 'base', 'small', 'medium', or 'large'.")

def process_all_media_files():
    if len(sys.argv) != 2:
        whisper_folder = os.getcwd()
        print("Defaulting to current directory - please enter another directory if you would like to transcribe from somewhere else.")
    else:
        whisper_folder = sys.argv[1]

    os.chdir(whisper_folder)

    media_files = [f for f in os.listdir(whisper_folder) if is_valid_file(f)]
    if not os.path.exists(whisper_folder) or len(media_files) == 0:
        print("No valid audio or video files found in the Whisper folder.")
        return

    print(f"Found {len(media_files)} valid file(s). Starting transcription process...")

    model_name = get_model_choice()
   
    print(f"Loading Whisper {model_name} model...")
    model = load_model(model_name)

    for media_file in media_files:
        transcribe_media(whisper_folder, media_file, model, model_name)
   
    print("All transcriptions completed.")

if __name__ == "__main__":
    process_all_media_files()
