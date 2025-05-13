import os
import queue
import sounddevice as sd
import soundfile as sf
import numpy as np
import whisper
from vosk import Model, KaldiRecognizer
from googletrans import Translator
from gtts import gTTS

# Settings
TRIGGER_WORD = "jarvis"
SAMPLE_RATE = 16000
MODEL_PATH = "models/vosk-model-small-en-us-0.15"
WHISPER_MODEL = "tiny"  # or 'base', 'small'

q = queue.Queue()

# 1. Audio stream callback
def callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(bytes(indata))

# 2. Detect trigger word using Vosk
def listen_for_trigger_and_save():
    print("🎧 Listening... Say something and end with 'Jarvis'")

    model = Model(MODEL_PATH)
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    audio_data = []

    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
        while True:
            data = q.get()
            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
                text = eval(result).get("text", "")
                print("📝 Heard:", text)
                audio_data.append(np.frombuffer(data, dtype=np.int16))
                if TRIGGER_WORD in text.lower():
                    print("✅ Trigger word detected.")
                    break
            else:
                audio_data.append(np.frombuffer(data, dtype=np.int16))

    full_audio = np.concatenate(audio_data)
    sf.write("input.wav", full_audio, SAMPLE_RATE)
    return "input.wav"

# 3. Transcribe with Whisper
def transcribe(file_path):
    model = whisper.load_model(WHISPER_MODEL)
    print("🧠 Transcribing...")
    result = model.transcribe(file_path)
    print("📝 Transcription:", result["text"])
    return result["text"]

# 4. Translate with googletrans
def translate_text(text):
    translator = Translator()
    if any(char in text for char in "가나다라마바사"):  # crude Korean check
        result = translator.translate(text, src='ko', dest='en')
        return result.text, 'en'
    else:
        result = translator.translate(text, src='en', dest='ko')
        return result.text, 'ko'

# 5. Speak with gTTS
def speak(text, lang):
    tts = gTTS(text=text, lang=lang)
    tts.save("output.mp3")
    os.system("afplay output.mp3" if os.name == "posix" else "start output.mp3")

# 6. Main loop
def main():
    audio_path = listen_for_trigger_and_save()
    original = transcribe(audio_path).replace(TRIGGER_WORD, "").strip()
    translated, lang = translate_text(original)
    print("🌐 Translated:", translated)
    speak(translated, lang)

if __name__ == "__main__":
    main()
