import speech_recognition as sr
import whisper
from deep_translator import GoogleTranslator
from gtts import gTTS
import os
import time
import requests
import tempfile
import deepl


AZURE_KEY = "7gib9HEapClSD02nShNgqunysUvMyVy5bMJuPvU58ccpGKt84ChYJQQJ99BEACNns7RXJ3w3AAAbACOG4XSJ"
AZURE_LOCATION = "koreacentral"
AZURE_ENDPOINT = "https://api.cognitive.microsofttranslator.com"

auth_key = "4305cb96-bc26-468a-8c7c-7ed035f9df33:fx"  # 여기에는 실제 키 입력
translator = deepl.Translator(auth_key)


def record_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎙️ Speak now...")
        audio = recognizer.listen(source)
        with open("input.wav", "wb") as f:
            f.write(audio.get_wav_data())
    return "input.wav"



def transcribe_audio(file_path):
    model = whisper.load_model("tiny")
    result = model.transcribe(file_path, task = "translate")
    return result["text"]



def translate_text(text, src_lang='EN', tgt_lang='KO'):
    result = translator.translate_text(text, source_lang=src_lang, target_lang=tgt_lang)
    return result

def azure_translate(text, src_lang="en", tgt_lang="ko"):
    print("🌐 Translating with Azure...")
    path = '/translate?api-version=3.0'
    params = f"&from={src_lang}&to={tgt_lang}"
    url = AZURE_ENDPOINT + path + params

    headers = {
        'Ocp-Apim-Subscription-Key': AZURE_KEY,
        'Ocp-Apim-Subscription-Region': AZURE_LOCATION,
        'Content-type': 'application/json',
    }

    body = [{'text': text}]
    response = requests.post(url, headers=headers, json=body)
    result = response.json()
    translated_text = result[0]['translations'][0]['text']
    return translated_text


def speak_text(text, lang='ko'):
    tts = gTTS(text=text, lang=lang)
    tts.save("output.mp3")
    os.system("start output.mp3" if os.name == "nt" else "afplay output.mp3")


def main():
    """audio_file = record_audio()"""

    audio_file = 'input02.wav'
    original_text = transcribe_audio(audio_file)
    print('\n\n')
    print("📝 Transcribed:", original_text)

    start_time = time.time()
    # Detect language
    if any(char in original_text for char in "가나다라마바사"):  # crude check for Korean
        translated = translate_text(original_text, src_lang='KO', tgt_lang='EN')
        """speak_text(translated, lang='en')"""

    else:
        translated =  translate_text(original_text)
        """speak_text(translated, lang='ko')"""
   
    end_time = time.time()

    print({end_time - start_time})
    print('\n\n')

    print("🌐 Translated: ", translated)
if __name__ == "__main__":
    main()
