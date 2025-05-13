import os
import threading
import queue
import sounddevice as sd
import soundfile as sf
import numpy as np
import whisper
from vosk import Model, KaldiRecognizer
from googletrans import Translator
from gtts import gTTS
import tkinter as tk

# === Configuration === #
TRIGGER_WORD = "jarvis"
SAMPLE_RATE = 16000
MODEL_PATH = "models/vosk-model-small-en-us-0.15"
# Use the smallest Whisper model for speed
WHISPER_MODEL = "tiny.en"

q = queue.Queue()
translator = Translator()

# === GUI Setup === #
class VoiceTranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("English-Korean Voice Translator")
        self.root.geometry("500x300")
        self.root.configure(bg='white')

        self.status_label = tk.Label(root, text="Click 'Start' to begin.", font=("Arial", 12), bg='white', fg='black')
        self.status_label.pack(pady=10)

        self.subtitle_label = tk.Label(root, text="", font=("Arial", 16), wraplength=480, bg='white', fg='black')
        self.subtitle_label.pack(pady=20)

        btn_frame = tk.Frame(root, bg='white')
        btn_frame.pack(pady=10)

        self.start_button = tk.Button(btn_frame, text="Start Listening", font=("Arial", 12), bg='lightgrey', fg='black',
                                       command=self.start_listening)
        self.start_button.grid(row=0, column=0, padx=5)

        self.stop_button = tk.Button(btn_frame, text="Stop Listening", font=("Arial", 12), bg='lightgrey', fg='black',
                                      command=self.stop_listening, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=5)

        self.running = False
        self.listen_thread = None

    def start_listening(self):
        if not self.running:
            self.running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text="🎤 Listening... Click 'Stop' when done.")
            # Clear previous subtitle
            self.subtitle_label.config(text="")
            self.listen_thread = threading.Thread(target=self.listen_for_trigger, daemon=True)
            self.listen_thread.start()

    def stop_listening(self):
        if self.running:
            self.running = False
            self.status_label.config(text="🛑 Stopping...")
            self.stop_button.config(state=tk.DISABLED)
            # Wait for listener to finish
            if self.listen_thread:
                self.listen_thread.join()
            # Start translation in background
            threading.Thread(target=self.translate_audio, daemon=True).start()

    def listen_for_trigger(self):
        model = Model(MODEL_PATH)
        recognizer = KaldiRecognizer(model, SAMPLE_RATE)
        audio_data = []

        def callback(indata, frames, time, status):
            if status:
                print(status)
            q.put(bytes(indata))

        with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000,
                               dtype='int16', channels=1, callback=callback):
            while self.running:
                data = q.get()
                if recognizer.AcceptWaveform(data):
                    res = recognizer.Result()
                    text = eval(res).get("text", "")
                    print("📝 Heard:", text)
                audio_data.append(np.frombuffer(data, dtype=np.int16))

        # Save captured audio
        if audio_data:
            full_audio = np.concatenate(audio_data)
            sf.write("input.wav", full_audio, SAMPLE_RATE)
        else:
            # No audio captured
            open("input.wav", 'wb').close()

    def translate_audio(self):
        self.status_label.config(text="⏳ Transcribing & Translating...")
        self.root.update_idletasks()

        # Transcribe with Whisper
        model = whisper.load_model(WHISPER_MODEL)
        result = model.transcribe("input.wav")
        original = result.get("text", "").replace(TRIGGER_WORD, "").strip()
        print("🧾 Transcribed:", original)

        # Translate
        if any(c in original for c in "가나다라마바사"):
            trans = translator.translate(original, src='ko', dest='en')
            lang = 'en'
        else:
            trans = translator.translate(original, src='en', dest='ko')
            lang = 'ko'
        translated = trans.text

        # Update UI
        self.subtitle_label.config(text=translated)
        self.status_label.config(text="✅ Done.")
        self.start_button.config(state=tk.NORMAL)
        self.root.update_idletasks()

        # Speak
        tts = gTTS(text=translated, lang=lang)
        tts.save("output.mp3")
        os.system("afplay output.mp3" if os.name == "posix" else "start output.mp3")

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceTranslatorApp(root)
    root.mainloop()
