import speech_recognition as sr
import whisper
from deep_translator import GoogleTranslator
from gtts import gTTS
import pyttsx3
import os
import time

# Load Whisper model once (suppress warnings)
import warnings
warnings.filterwarnings("ignore")
model = whisper.load_model("base")

# Keyword to stop
STOP_KEYWORD = "stop translation"

# Text-to-Speech engine
engine = pyttsx3.init()

def record_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n🎙️ Listening...")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=None)
            return audio
        except sr.WaitTimeoutError:
            return None

def transcribe_audio(audio):
    if not audio:
        return None
        
    # Save temporary audio file
    with open("temp_audio.wav", "wb") as f:
        f.write(audio.get_wav_data())
    
    # Transcribe
    try:
        result = model.transcribe("temp_audio.wav")
        text = result["text"].strip()
        
        # Clean up
        os.remove("temp_audio.wav")
        
        if text:
            print(f"\n💬 You said: {text}")
            return text
    except Exception as e:
        print("\n❌ Failed to transcribe audio")
    
    return None

def translate_text(text, src_lang='en', tgt_lang='ko'):
    try:
        translated = GoogleTranslator(source=src_lang, target=tgt_lang).translate(text)
        if translated:
            print(f"🌐 Translation: {translated}")
            return translated
    except Exception:
        print("\n❌ Translation failed")
    return None

def speak_text(text, lang='ko'):
    if not text:
        return
        
    try:
        tts = gTTS(text=text, lang=lang)
        tts.save("output.mp3")
        os.system("afplay output.mp3" if os.name != "nt" else "start output.mp3")
        os.remove("output.mp3")
    except Exception:
        try:
            engine.say(text)
            engine.runAndWait()
        except:
            print("\n❌ Failed to speak translation")

def detect_language(text):
    return "ko" if any(char in text for char in "가나다라마바사아자차카타파하") else "en"

def main():
    print("\n🔄 Real-time Voice Translator")
    print("─" * 30)
    print("• Speak naturally in English or Korean")
    print("• Say 'stop translation' to exit")
    print("• Press Ctrl+C to force quit")
    print("─" * 30)

    while True:
        try:
            # Record
            audio = record_audio()
            if not audio:
                continue

            # Transcribe
            text = transcribe_audio(audio)
            if not text:
                continue

            # Check for stop command
            if STOP_KEYWORD in text.lower():
                print("\n👋 Goodbye!")
                break

            # Translate
            src_lang = detect_language(text)
            tgt_lang = 'en' if src_lang == 'ko' else 'ko'
            translated = translate_text(text, src_lang, tgt_lang)

            # Speak
            if translated:
                speak_text(translated, lang=tgt_lang)
                print("─" * 30)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            continue

if __name__ == "__main__":
    main()
