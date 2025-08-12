#!/usr/bin/env python3

import numpy as np
import sounddevice as sd
import mlx_whisper
from mlx_whisper import load_models
import torch
# Removed complex VAD imports - using simple energy-based detection
import threading
import time
import queue
import sys
from collections import deque
from datetime import datetime, timedelta

class ActiveAssistant:
    def __init__(self, trigger_words=None, callback=None, sample_rate=16000, verbose=False, 
                 model="mlx-community/whisper-large-v3-turbo", vad_threshold=0.008, buffer_seconds=15, chunk_seconds=0.25):
        """
        Initialize active assistant with volume-based VAD and smart phrase ending.
        """
        self.callback = callback
        self.sample_rate = sample_rate
        self.verbose = verbose
        self.vad_threshold = vad_threshold
        self.buffer_seconds = buffer_seconds
        self.chunk_seconds = chunk_seconds
        
        # Default trigger words
        self.trigger_words = trigger_words or [
            "jeeves", "help", "music", "lights", "time", "weather", 
            "turn on", "turn off", "play", "stop", "assistant"
        ]
        
        if verbose:
            print(f"Loading models... Trigger words: {self.trigger_words}")
            print(f"Using volume-based VAD (threshold: {vad_threshold})")
        
        # Load models
        self.model_name = model
        self.whisper_model = load_models.load_model(path_or_hf_repo=self.model_name)
        torch.set_num_threads(1)
        
        if verbose:
            print("✅ Models ready")
        
        # Audio processing state
        self.block_size = int(sample_rate * chunk_seconds)
        self.max_samples = int(sample_rate * buffer_seconds)
        self.assistant_speaking = False
        self.in_speech = False
        self.silence_end_seconds = 0.9  # end phrase after ~0.9s of silence
        self.last_transcription = ""
        self.last_word_count = 0
        self.silence_start_time = 0
        self.ending_punctuation = {'.', '!', '?'}
        
        # Auto-trigger window after assistant speaks
        self.auto_trigger_window = 12.0  # seconds (slightly longer)
        self.auto_trigger_end_time = 0
        
        # Forced listening window (used by wait_for_response)
        self.force_listen_until = 0.0
        
        # De-duplication for emitted phrases
        self.last_emitted_transcription = ""
        self.last_emitted_time = 0.0
        
        # Speech history buffer (6 minutes)
        self.speech_history = deque()
        self.history_duration = timedelta(minutes=6)
        
        # Audio queue
        self.audio_q = queue.Queue()

    def arm_next_phrase(self, timeout: float = 12.0):
        """Arm the assistant to treat the next completed user phrase as a reply without blocking.
        Keeps normal listening; just widens the auto-trigger window and allows wake-word bypass.
        """
        # Ensure we're actively listening
        self.assistant_speaking = False
        now = time.time()
        # Open both auto-trigger and forced-listen windows
        self.auto_trigger_end_time = max(self.auto_trigger_end_time, now + float(timeout))
        self.force_listen_until = max(self.force_listen_until, now + float(timeout))
        # Clear de-duplication so we emit promptly
        self.last_emitted_transcription = ""
        self.last_emitted_time = 0.0
    
    def is_volume_above_threshold(self, buf):
        """Simple volume-based detection"""
        if len(buf) < 256:
            return False
        
        # Calculate RMS energy (volume)
        rms_energy = np.sqrt(np.mean(buf**2))
        return rms_energy > self.vad_threshold
    
    def should_end_phrase(self, transcription):
        """Determine if phrase should end based on word count and punctuation"""
        if not transcription:
            return False
            
        current_word_count = len(transcription.split())
        has_new_words = current_word_count > self.last_word_count
        
        if has_new_words:
            self.last_word_count = current_word_count
            self.silence_start_time = time.time()  # Reset silence timer
            return False
        
        # No new words - check timing and punctuation
        silence_duration = time.time() - self.silence_start_time
        has_ending_punctuation = any(transcription.rstrip().endswith(p) for p in self.ending_punctuation)
        
        if has_ending_punctuation and silence_duration >= 1.0:
            return True
        elif silence_duration >= 3.0:
            return True
            
        return False
    
    def transcribe_audio(self, audio_data):
        """Transcribe audio with mlx-whisper model"""
        result = mlx_whisper.transcribe(
            audio_data,
            path_or_hf_repo=self.model_name,
            language=None,
            word_timestamps=False,
        )
        return result["text"].strip()
    
    def check_trigger_words(self, transcription):
        """Check if transcription contains any trigger words"""
        text_lower = transcription.lower()
        text_lower = text_lower.replace("jeeps", "jeeves").replace("jeebs", "jeeves")
        matched_words = []
        
        for trigger in self.trigger_words:
            if trigger.lower() in text_lower:
                matched_words.append(trigger)
        
        return len(matched_words) > 0, matched_words
    
    def add_to_history(self, transcription):
        """Add transcription to history with timestamp"""
        timestamp = datetime.now()
        self.speech_history.append({
            'timestamp': timestamp,
            'transcription': transcription
        })
        
        # Remove entries older than 6 minutes
        cutoff_time = timestamp - self.history_duration
        while self.speech_history and self.speech_history[0]['timestamp'] < cutoff_time:
            self.speech_history.popleft()
    
    def get_recent_history(self):
        """Get all speech from the past 6 minutes with timestamps"""
        return list(self.speech_history)
    
    def clear_history(self):
        """Clear the conversation history"""
        self.speech_history.clear()
        if self.verbose:
            print("🗑️ Conversation history cleared")
    
    # Removed blocking wait_for_response in favor of non-blocking arm_next_phrase
    
    def on_trigger_detected(self, transcription, matched_words):
        """Handle detection of trigger words"""
        if self.verbose:
            print(f"🎯 TRIGGERED: '{transcription}' - Words: {matched_words}")
        
        # Update de-duplication state
        self.last_emitted_transcription = transcription
        self.last_emitted_time = time.time()
        
        if self.callback:
            history = self.get_recent_history()
            # Call callback directly - no threading complications
            self.callback(transcription, history)
    
    def set_assistant_speaking(self, speaking):
        """Set whether the assistant is currently speaking"""
        if self.verbose:
            if speaking:
                print("🔇 Assistant speaking - pausing audio processing")
            else:
                print("🎤 Assistant finished - resuming audio processing")
        self.assistant_speaking = speaking
        
        # Start auto-trigger window when assistant finishes speaking
        if not speaking:
            self.auto_trigger_end_time = time.time() + self.auto_trigger_window
    
    def audio_callback(self, indata, frames, _time, status):
        """Audio stream callback"""
        if status:
            print(status, file=sys.stderr)
        
        # Simple: if assistant is speaking, don't process audio
        if self.assistant_speaking:
            return
            
        audio_chunk = indata[:, 0].copy()
        self.audio_q.put_nowait(audio_chunk)
    
    def process_audio_loop(self):
        """Main audio processing loop with immediate phrase start and silence-based ending"""
        phrase_buffer = np.zeros(0, dtype=np.float32)
        last_activity_time = 0.0
        
        try:
            while True:
                chunk = self.audio_q.get()

                # Fast-forward if there's backlog to reduce latency
                if self.audio_q.qsize() > 8:
                    try:
                        while self.audio_q.qsize() > 1:
                            chunk = self.audio_q.get_nowait()
                    except queue.Empty:
                        pass

                # Skip processing if assistant is speaking
                if self.assistant_speaking:
                    continue

                # Compute volume on current chunk only for immediate reaction
                volume_detected = self.is_volume_above_threshold(chunk)

                now = time.time()

                if volume_detected:
                    if not self.in_speech:
                        if self.verbose:
                            timestamp = time.strftime("%H:%M:%S", time.localtime())
                            print(f"<PHRASE_START:{timestamp}>")
                        self.in_speech = True
                        phrase_buffer = np.zeros(0, dtype=np.float32)
                    # Append chunk and mark activity
                    phrase_buffer = np.concatenate([phrase_buffer, chunk])
                    # Limit phrase buffer to max_samples to cap length
                    if len(phrase_buffer) > self.max_samples:
                        phrase_buffer = phrase_buffer[-self.max_samples:]
                    last_activity_time = now
                else:
                    # No volume in this chunk
                    if self.in_speech:
                        # Append low-energy chunk as well to capture tails
                        phrase_buffer = np.concatenate([phrase_buffer, chunk])
                        if len(phrase_buffer) > self.max_samples:
                            phrase_buffer = phrase_buffer[-self.max_samples:]
                        # End phrase after sustained silence
                        if (now - last_activity_time) >= self.silence_end_seconds:
                            if self.verbose:
                                timestamp = time.strftime("%H:%M:%S", time.localtime())
                                print(f"<PHRASE_END:{timestamp}>")

                            # Transcribe the captured phrase once
                            try:
                                transcription = self.transcribe_audio(phrase_buffer)
                            except Exception as e:
                                transcription = ""
                                if self.verbose:
                                    print(f"Transcription error: {e}")

                            self.last_transcription = transcription

                            if transcription:
                                if self.verbose:
                                    print(f"Heard: '{transcription}'")
                                self.add_to_history(transcription)
                                triggered, matched_words = self.check_trigger_words(transcription)

                                in_auto_window = now < self.auto_trigger_end_time
                                in_force_window = now < self.force_listen_until

                                should_emit = True
                                if (transcription == self.last_emitted_transcription
                                    and (now - self.last_emitted_time) < 1.5):
                                    should_emit = False

                                if (triggered or in_auto_window or in_force_window) and should_emit:
                                    self.on_trigger_detected(transcription, matched_words)

                            # Reset state after phrase
                            self.in_speech = False
                            self.last_transcription = ""
                            self.last_word_count = 0
                            phrase_buffer = np.zeros(0, dtype=np.float32)
                            
        except Exception as e:
            if self.verbose:
                print(f"Audio processing error: {e}")
    
    def start_listening(self):
        """Start listening for all speech"""
        if self.verbose:
            print("🎤 Active listening started...")
        
        # Start audio processing thread
        audio_thread = threading.Thread(target=self.process_audio_loop, daemon=True)
        audio_thread.start()
        
        with sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=1,
            dtype="float32",
            callback=self.audio_callback,
        ):
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                if self.verbose:
                    print("\n🛑 Stopped listening")

def main():
    """Demo the active assistant"""
    def demo_callback(transcription, history):
        print(f"🚨 TRIGGER DETECTED: '{transcription}'")
        print(f"📜 CONVERSATION HISTORY ({len(history)} entries):")
        for entry in history:
            timestamp_str = entry['timestamp'].strftime("%H:%M:%S")
            print(f"   [{timestamp_str}] {entry['transcription']}")
    
    detector = ActiveAssistant(callback=demo_callback, verbose=True)
    print("🎤 Using volume-based VAD with smart phrase ending")
    print("Speech ends after 1s if ending punctuation detected, otherwise 3s")
    print("Try saying trigger words like: 'Jeeves', 'help', 'music', 'turn on lights'")
    try:
        detector.start_listening()
    except KeyboardInterrupt:
        print("\nDemo stopped.")

if __name__ == "__main__":
    main()