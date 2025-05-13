import speech_recognition as sr
import whisper
from deep_translator import GoogleTranslator
from gtts import gTTS
import os


def record_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎙️ Speak now...")
        audio = recognizer.listen(source)
        with open("input.wav", "wb") as f:
            f.write(audio.get_wav_data())
    return "input.wav"



def transcribe_audio(file_path):
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    print("📝 Transcribed:", result["text"])
    return result["text"]



def translate_text(text, src_lang='en', tgt_lang='ko'):
    translator = GoogleTranslator()
    translated = translator.translate(text, src=src_lang, dest=tgt_lang)
    print("🌐 Translated:", translated.text)
    return translated.text


def speak_text(text, lang='ko'):
    tts = gTTS(text=text, lang=lang)
    tts.save("output.mp3")
    os.system("start output.mp3" if os.name == "nt" else "afplay output.mp3")


def main():
    audio_file = record_audio()
    original_text = transcribe_audio(audio_file)
    
    # Detect language
    if any(char in original_text for char in "가나다라마바사"):  # crude check for Korean
        translated = translate_text(original_text, src_lang='ko', tgt_lang='en')
        speak_text(translated, lang='en')
    else:
        translated = translate_text(original_text, src_lang='en', tgt_lang='ko')
        speak_text(translated, lang='ko')

if __name__ == "__main__":
    main()
