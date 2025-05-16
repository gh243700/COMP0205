import os
import queue
import sounddevice as sd
import soundfile as sf
import numpy as np
import whisper
from vosk import Model, KaldiRecognizer
from deep_translator import GoogleTranslator
from gtts import gTTS
import json

# Settings
TRIGGER_WORD = "stop"
SAMPLE_RATE = 16000
MODEL_PATH = "models/vosk-model-small-en-us-0.15"
WHISPER_MODEL = "tiny"  # or 'base', 'small', etc.

q = queue.Queue()

# 1. Audio stream callback
def callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(bytes(indata))

# 2. Listen for trigger and save audio to file
def listen_for_trigger_and_save():
    print("🎧 Listening... Speak and end with 'STOP'")

    model = Model(MODEL_PATH)
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    audio_data = []

    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
        while True:
            data = q.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
                print("📝 Heard:", text)
                audio_data.append(np.frombuffer(data, dtype=np.int16))
                if TRIGGER_WORD in text.lower():
                    print("✅ Trigger word detected.")
                    break
            else:
                audio_data.append(np.frombuffer(data, dtype=np.int16))

    full_audio = np.concatenate(audio_data)
    sf.write("input.wav", full_audio, SAMPLE_RATE)
    duration_seconds = len(full_audio) / SAMPLE_RATE
    print(f"⏱️ Recording duration: {duration_seconds:.2f} seconds")
    return "input.wav",duration_seconds

# 3. Transcribe audio with Whisper
def transcribe(file_path):
    model = whisper.load_model(WHISPER_MODEL)
    print("🧠 Transcribing...")
    result = model.transcribe(file_path)
    print("📝 Transcription:", result["text"])
    return result["text"]

# 4. Translate using deep-translator
def translate_text(text):
    if any(char in text for char in "가나다라마바사"):  # crude Korean check
        translated = GoogleTranslator(source='ko', target='en').translate(text)
        return translated, 'en'
    else:
        translated = GoogleTranslator(source='en', target='ko').translate(text)
        return translated, 'ko'

# 5. Speak translated text
def speak(text, lang):
    tts = gTTS(text=text, lang=lang)
    tts.save("output.mp3")
    os.system("afplay output.mp3" if os.name == "posix" else "start output.mp3")

# 6. Main logic
def main():
    audio_path,duration = listen_for_trigger_and_save()
    print(f"Recording duration (in main): {duration:.2f} seconds")  
    original = transcribe(audio_path)

    # Remove everything from "stop" onward
    text_lower = original.lower()
    cut_index = text_lower.find(TRIGGER_WORD)
    if cut_index != -1:
        original = original[:cut_index].strip()

    translated, lang = translate_text(original)
    print("🌐 Translated:", translated)
    speak(translated, lang)

if __name__ == "__main__":
    main()
