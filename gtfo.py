import os
import queue
import sounddevice as sd
import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel,BatchedInferencePipeline
import string
from vosk import Model, KaldiRecognizer
from deep_translator import GoogleTranslator
from gtts import gTTS
import json
import threading
from threading import Thread
import concurrent.futures

# Settings
    

TRIGGER_WORD = "stop"
SAMPLE_RATE = 16000
MODEL_PATH = "models/vosk-model-small-en-us-0.15"
WHISPER_MODEL = "turbo"
MODEL = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
batched_model = BatchedInferencePipeline(model=MODEL)


q = queue.Queue()

BUFFER_SIZE = 1024
buffer = [('', 0, '', 0) for _ in range(BUFFER_SIZE)]  # (text, flag, lang, duration_time)
b_index = 0
b_ptr = 0
lock = threading.Lock()

result_queue = queue.Queue()

# 1. Audio stream callback
def callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(bytes(indata))

# 2. Listen for trigger and save audio
def listen_for_trigger_and_save():
    print("🎧 Listening... Speak and end with 'STOP'")

    model = Model(MODEL_PATH)
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    audio_data = []
    total_audio_data = []
    num = 0
    
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
        
        st = ''
        while True:
            data = q.get()
            recognizer.AcceptWaveform(data)
            result = json.loads(recognizer.Result())
            text = result.get("text", "")
            if text == "":
                if (st != ''):
                    audio = np.concatenate(audio_data)
                    #sf.write("input"+str(num)+".wav", audio, SAMPLE_RATE)
                    duration_seconds = len(audio) / SAMPLE_RATE
                    #audio_data = []
                    add_item(st, duration_seconds)
                    st = ''
                    num += 1

                continue
            
            st += text
            total_audio_data.append(np.frombuffer(data, dtype=np.int16))
            audio_data.append(np.frombuffer(data, dtype=np.int16))
            if TRIGGER_WORD in text.lower():
                print("✅ Trigger word detected.")
                break
            
        #full_audio = np.concatenate(total_audio_data)
        #sf.write("input.wav", full_audio, SAMPLE_RATE)
        #duration_seconds = len(full_audio) / SAMPLE_RATE
        #print(f"⏱️ Recording duration: {duration_seconds:.2f} seconds")
        #result_queue.put(("input.wav", duration_seconds))

def add_item(audio_path, duration_seconds):
        global b_index
        while (buffer[b_index][1] != 0):
            continue

        buffer[b_index] = (audio_path, 1, '', duration_seconds)
        b_index = (b_index + 1) % BUFFER_SIZE
    

def translate_text(i):
    global MODEL
    text, flag, lang, duration_seconds = buffer[i]
    #with lock:
    #    segments, info = batched_model.transcribe(audio_path, beam_size=5, batch_size=16)
    
    translated = ''
    if any(char in text for char in "가나다라마바사"):  # crude Korean check
        translated = GoogleTranslator(source='ko', target='en').translate(text)
        lang = 'en'
    else:
        translated = GoogleTranslator(source='en', target='ko').translate(text)
        lang = 'ko'

    buffer[i] = (translated, 2, lang, duration_seconds)

def transRoutine():
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        i = 0
        
        while True:
            while (buffer[i][1] != 1):
                continue 
            executor.submit(translate_text, i)
            i = (i + 1) % BUFFER_SIZE

def speak(text, lang):
    tts = gTTS(text=text, lang=lang)
    tts.save("output.mp3")
    os.system("afplay output.mp3" if os.name == "posix" else "start output.mp3")

def use_item_routine():
    global b_ptr
    while True:
        while (buffer[b_ptr][1] != 2):
            continue
        
        text, flag, lang, duration_seconds = buffer[b_ptr]
        print(f"🔈 Speaking: {text}")
        #speak(text, lang)
        buffer[b_ptr] = ('', 0, '', 0)
        b_ptr = (b_ptr + 1) % BUFFER_SIZE

# 6. Main logic
def main():

    
    producer_thread = Thread(target=listen_for_trigger_and_save)
    translator_thread = Thread(target=transRoutine)
    speaker_thread = Thread(target=use_item_routine)

    producer_thread.start()
    translator_thread.start()
    speaker_thread.start()

    producer_thread.join()
    audio_path, duration = result_queue.get()

    print(f"🎤 Audio saved: {audio_path}")
    print(f"⏱️ Total recording duration: {duration:.2f} seconds")

if __name__ == "__main__":
    main()
