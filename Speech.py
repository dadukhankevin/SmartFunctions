import sounddevice as sd
import requests
import base64
import numpy as np
import threading
import re

API_KEY = 'AIzaSyCx79POrrR5AgCPMTQRW0gH5ty2z4k0iug'
pipeline = None  # Lazy initialization

def clean_text(text):
    text = text.replace("*", "")
    return text

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]

def speak_audio(audio_content):
    """Helper to play audio content"""
    audio_array = np.frombuffer(audio_content, dtype=np.int16).astype(np.float32) / 32768.0
    sd.play(audio_array, 24000, blocksize=2048)
    sd.wait()

def speak(text):
    text = clean_text(text)
    print("Text:", text)
    
    sentences = split_into_sentences(text)
    if not sentences:
        return
    
    try:
        url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={API_KEY}"
        
        # Speak first sentence immediately
        first_data = {
            "input": {"text": sentences[0]},
            "voice": {"languageCode": "en-GB", "name": "en-GB-Chirp3-HD-Enceladus"},
            "audioConfig": {
                "audioEncoding": "LINEAR16",
                "effectsProfileId": ["large-home-entertainment-class-device"],
                "speakingRate": 1
            }
        }
        
        response = requests.post(url, json=first_data)
        first_audio = base64.b64decode(response.json()['audioContent'])
        
        # Load remaining sentences in background
        remaining_audio = []
        def load_remaining():
            for sentence in sentences[1:]:
                data = {
                    "input": {"text": sentence},
                    "voice": {"languageCode": "en-GB", "name": "en-GB-Chirp3-HD-Enceladus"},
                    "audioConfig": {
                        "audioEncoding": "LINEAR16",
                        "effectsProfileId": ["large-home-entertainment-class-device"],
                        "speakingRate": 1
                    }
                }
                resp = requests.post(url, json=data)
                remaining_audio.append(base64.b64decode(resp.json()['audioContent']))
        
        # Start loading in background
        thread = threading.Thread(target=load_remaining)
        thread.start()
        
        # Play first sentence
        speak_audio(first_audio)
        
        # Wait for background loading and play remaining
        thread.join()
        for audio in remaining_audio:
            speak_audio(audio)
            
        print("Google TTS")
        
    except:
        print("Kokoro fallback")
        global pipeline
        if pipeline is None:
            from kokoro import KPipeline
            pipeline = KPipeline(lang_code='b', repo_id='hexgrad/Kokoro-82M')
        generator = pipeline(text, voice='bm_daniel')
        for _, _, audio in generator:
            sd.play(audio, 24000)
            sd.wait()

if __name__ == "__main__":
    input("Start: ")
    speak("Hello, this is a test of my vocal system.")