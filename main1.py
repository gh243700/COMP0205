import speech_recognition as sr
import whisper
from deep_translator import GoogleTranslator
from gtts import gTTS
import os

# Load Whisper model once
model = whisper.load_model("base")

# Keyword to stop the loop
STOP_KEYWORD = "stop translation"

def record_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎙️ Speak now...")
        audio = recognizer.listen(source)
        with open("input.wav", "wb") as f:
            f.write(audio.get_wav_data())
    return "input.wav"

def transcribe_audio(file_path):
    result = model.transcribe(file_path)
    text = result["text"].strip()
    print("📝 Transcribed:", text)
    return text

def translate_text(text, src_lang='en', tgt_lang='ko'):
    translated = GoogleTranslator(source=src_lang, target=tgt_lang).translate(text)
    print("🌐 Translated:", translated)
    return translated

def speak_text(text, lang='ko'):
    tts = gTTS(text=text, lang=lang)
    tts.save("output.mp3")
    os.system("start output.mp3" if os.name == "nt" else "afplay output.mp3")

def detect_language(text):
    return "ko" if any(char in text for char in "가나다라마바사") else "en"

def main_loop():
    print("🔁 Starting real-time translation. Say 'stop translation' to exit.")
    while True:
        audio_file = record_audio()
        original_text = transcribe_audio(audio_file)

        if STOP_KEYWORD in original_text.lower():
            print("🛑 Stop keyword detected. Exiting...")
            break

        src_lang = detect_language(original_text)
        tgt_lang = 'en' if src_lang == 'ko' else 'ko'

        translated_text = translate_text(original_text, src_lang, tgt_lang)
        speak_text(translated_text, lang=tgt_lang)

if __name__ == "__main__":
    main_loop()
