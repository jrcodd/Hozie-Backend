import sounddevice as sd
import numpy as np
from datetime import datetime
import requests
import os
import json
import queue
import vosk
from typing import Optional, Tuple
from dotenv import load_dotenv
from Session import Session  

load_dotenv()

class HozieVoiceSynthesizer:
    def __init__(self, api_key: str, base_url: str = "https://api.async.ai", voice_id: str = None,
                 vosk_model_path: str = None):
        """
        Initialize the AsyncFlow TTS voice synthesizer with Vosk speech recognition.
        
        Args:
            api_key (str): API key for authentication
            base_url (str): Base URL for the API
            voice_id (str): Default voice ID to use
            vosk_model_path (str): Path to Vosk model directory
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.voice_id = voice_id or "e0f39dc4-f691-4e78-bba5-5c636692cc04"
        self.endpoint = f"{self.base_url}/text_to_speech/streaming"
        
        # Default output format settings
        self.output_format = {
            "container": "raw",
            "encoding": "pcm_f32le",
            "sample_rate": 44100
        }
        
        # Audio recording settings
        self.sample_rate = 16000  # Vosk typically works best with 16kHz
        self.block_size = 8000
        self.audio_queue = queue.Queue()
        
        print(f"Initializing AsyncFlow TTS voice synthesizer...")
        print(f"API Endpoint: {self.endpoint}")
        print(f"Default Voice ID: {self.voice_id}")

        self.is_initialized = self._test_connection()

        if self.is_initialized:
            print("✓ AsyncFlow TTS initialized successfully!")
        else:
            print("✗ Failed to initialize AsyncFlow TTS")
            return
        
        print("\nInitializing Vosk speech recognition...")
        if not vosk_model_path:
            print("✗ No Vosk model found. Please download a model from:")
            print("  https://alphacephei.com/vosk/models")
            print("  Example: wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip")
            print("  Then unzip and provide the path")
            self.vosk_model = None
            return
        
        try:
            self.vosk_model = vosk.Model(vosk_model_path)
            self.vosk_rec = vosk.KaldiRecognizer(self.vosk_model, self.sample_rate)
            print(f"✓ Vosk initialized with model: {vosk_model_path}")
        except Exception as e:
            print(f"✗ Failed to initialize Vosk: {e}")
            self.vosk_model = None
        
        self.session = Session()
        print("✓ Brain session initialized")
    
    def _test_connection(self) -> bool:
        """
        Test the API connection with a simple request.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        
        try:
            headers = {
                "X-Api-Key": self.api_key,
                "Content-Type": "application/json"
            }
            
            data = {
                "model_id": "asyncflow_v2.0",
                "transcript": "Test",
                "voice": {
                    "mode": "id",
                    "id": self.voice_id
                },
                "output_format": self.output_format
            }
            
            response = requests.post(
                self.endpoint, 
                json=data, 
                headers=headers, 
                stream=True,
                timeout=5
            )
            
            if response.status_code == 200:
                for _ in response.iter_content(chunk_size=1024):
                    pass
                return True
            else:
                print(f"API Error: {response.status_code}")
                if response.headers.get('content-type') == 'application/json':
                    error_data = response.json()
                    print(f"Error details: {error_data}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"Connection error: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False
    
    def _decode_audio_stream(self, audio_data: bytes, encoding: str, sample_rate: int) -> Tuple[np.ndarray, int]:
        """
        Decode raw audio stream to numpy array.
        
        Args:
            audio_data (bytes): Raw audio data from the stream
            encoding (str): Audio encoding format (e.g., "pcm_f32le", "pcm_s16le")
            sample_rate (int): Sample rate of the audio
        
        Returns:
            Tuple[np.ndarray, int]: Tuple containing the audio array and sample rate
        """

        if encoding == "pcm_f32le":
            audio_array = np.frombuffer(audio_data, dtype='<f4')
        elif encoding == "pcm_s16le":
            audio_array = np.frombuffer(audio_data, dtype='<i2').astype(np.float32) / 32768.0
        else:
            raise ValueError(f"Unsupported encoding: {encoding}")
        
        return audio_array, sample_rate
    
    def speak(self, text: str, voice_id: Optional[str] = None, speed: float = 1.0) -> Optional[float]:
        """
        Convert text to speech using AsyncFlow API and play it.
        
        Args:
            text (str): Text to convert to speech
            voice_id (str): Optional voice ID to use for synthesis
            speed (float): Speed factor for speech synthesis (default is 1.0)
        
        Returns:
            Optional[float]: Time taken to generate speech in seconds, or None if failed
        """

        if not self.is_initialized:
            print("TTS not initialized properly")
            return None
        
        if not text or not isinstance(text, str):
            print("Invalid text input")
            return None
        
        voice_id = voice_id or self.voice_id
        
        try:
            start_time = datetime.now()
            
            headers = {
                "X-Api-Key": self.api_key,
                "Content-Type": "application/json"
            }
            
            data = {
                "model_id": "asyncflow_v2.0",
                "transcript": text,
                "voice": {
                    "mode": "id",
                    "id": voice_id
                },
                "output_format": self.output_format
            }
            
            response = requests.post(
                self.endpoint,
                json=data,
                headers=headers,
                stream=True,
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"API Error: {response.status_code}")
                if response.headers.get('content-type') == 'application/json':
                    error_data = response.json()
                    print(f"Error details: {error_data}")
                return None
            
            audio_chunks = []
            for chunk in response.iter_content(chunk_size=4096):
                if chunk:
                    if b"--ERROR:QUOTA_EXCEEDED--" in chunk:
                        print("Error: Quota exceeded during stream")
                        return None
                    audio_chunks.append(chunk)
            
            audio_data = b''.join(audio_chunks)
            generation_time = (datetime.now() - start_time).total_seconds()
            
            audio_array, sample_rate = self._decode_audio_stream(
                audio_data, 
                self.output_format["encoding"],
                self.output_format["sample_rate"]
            )
            
            if audio_array.ndim == 1:
                audio_array = audio_array.reshape(-1, 1)
            
            sd.play(audio_array, sample_rate)
            sd.wait()
            
            return generation_time
            
        except Exception as e:
            print(f"Error during speech synthesis: {e}")
            return None
    
    def stop(self):
        """
        Stop any currently playing audio.
        """
        
        sd.stop()
    
    def audio_callback(self, indata, status):
        """
        Callback for audio recording.
        """
        
        if status:
            print(f"Audio callback status: {status}")
        self.audio_queue.put(bytes(indata))
    
    def listen_for_speech(self, timeout: float = 10.0) -> Optional[str]:
        """
        Listen for speech and return the recognized text.
        
        Args:
            timeout: Maximum time to listen in seconds
            
        Returns:
            Recognized text or None if nothing detected
        """
        if not self.vosk_model:
            print("Vosk not initialized")
            return None
        
        print("\nListening... (speak now)")
        
        # Clear the queue
        while not self.audio_queue.empty():
            self.audio_queue.get()
        
        # Start recording
        stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            dtype='int16',
            channels=1,
            callback=self.audio_callback
        )
        
        recognized_text = ""
        silence_duration = 0
        max_silence = 2.0  # Stop after 2 seconds of silence
        start_time = datetime.now()
        
        try:
            with stream:
                while True:
                    if (datetime.now() - start_time).total_seconds() > timeout:
                        print("\nTimeout reached")
                        break
                    
                    try:
                        data = self.audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    
                    if self.vosk_rec.AcceptWaveform(data):
                        result = json.loads(self.vosk_rec.Result())
                        if result.get('text'):
                            recognized_text = result['text']
                            print(f"\nHeard: {recognized_text}")
                            break
                    else:
                        partial = json.loads(self.vosk_rec.PartialResult())
                        if partial.get('partial'):
                            print(f"\r{partial['partial']}...", end='', flush=True)
                            silence_duration = 0
                        else:
                            silence_duration += 0.1
                            
                    if silence_duration > max_silence and recognized_text:
                        break
        
        except Exception as e:
            print(f"\nError during speech recognition: {e}")
        
        final_result = json.loads(self.vosk_rec.FinalResult())
        if final_result.get('text') and not recognized_text:
            recognized_text = final_result['text']
            print(f"\nFinal: {recognized_text}")

        return recognized_text if recognized_text else None
    
    def voice_conversation(self) -> None:
        """
        Run a voice-based conversation loop.
        Listen for speech, send to brain, speak the response.
        """
        if not self.vosk_model:
            print("Cannot start voice conversation - Vosk not initialized")
            return
        
        print("\n" + "="*60)
        print("Voice Conversation Mode")
        print("="*60)
        print("Speak naturally and I'll respond!")
        print("Say 'goodbye' or press Ctrl+C to exit\n")
        
        try:
            while True:
                user_input = self.listen_for_speech()
                
                if not user_input:
                    print("Didn't catch that. Please try again.")
                    continue
                
                # Check for exit commands
                if any(word in user_input.lower() for word in ['goodbye', 'bye', 'exit', 'quit']):
                    print("\nGoodbye!")
                    self.speak("Goodbye!")
                    break

                print(f"\nProcessing: '{user_input}'")

                # Get response from brain
                start_time = datetime.now()
                try:
                    response = self.session.answer(user_input)
                    brain_time = (datetime.now() - start_time).total_seconds()
                    print(f"Brain responded in {brain_time:.2f}s")
                except Exception as e:
                    print(f"Brain error: {e}")
                    response = "I'm sorry, I encountered an error processing that."
                
                # Speak the response
                response = response.replace("haha", "").replace("Aight", "ight").replace("*", "").replace("ya", "yeuh")
                print(f"\nSpeaking: {response}")
                generation_time = self.speak(response)
                
                if generation_time:
                    print(f"✓ Speech generated in {generation_time:.2f}s")
                
                print("\n" + "-"*40)
                
        except KeyboardInterrupt:
            print("\n\nVoice conversation ended")
            self.stop()
        except Exception as e:
            print(f"\nError in voice conversation: {e}")
    
    def interactive_session(self) -> None:
        """
        Enhanced interactive session with voice option.
        """
        print("\n" + "="*60)
        print("Interactive AsyncFlow TTS Session")
        print("="*60)
        print(f"Voice ID: {self.voice_id}")
        print(f"Output Format: {self.output_format['encoding']} @ {self.output_format['sample_rate']}Hz")
        
        if self.vosk_model:
            print("\n✓ Voice recognition available!")
        
        print("\nModes:")
        print("  1. Text mode (type text)")
        print("  2. Voice mode (speak to interact)" + (" [Available]" if self.vosk_model else " [Unavailable - no Vosk model]"))
        print("\nCommands:")
        print("  /mode <text|voice>: Switch input mode")
        print("  /voice <id>: Change voice ID")
        print("  /format <encoding> <sample_rate>: Change output format")
        print("  /help: Show commands")
        print("  Ctrl+C: Exit\n")
        
        mode = "text"
        
        try:
            while True:
                if mode == "voice" and self.vosk_model:
                    self.voice_conversation()
                    # After voice conversation, return to text mode
                    mode = "text"  
                    print("\nReturned to text mode. Type /mode voice to start voice mode again.")
                else:
                    text = input(">>> ")
                    
                    if not text.strip():
                        continue
                    
                    if text.startswith('/mode '):
                        new_mode = text.split()[1].lower()
                        if new_mode == "voice":
                            if self.vosk_model:
                                mode = "voice"
                                continue
                            else:
                                print("Voice mode unavailable - Vosk not initialized")
                        elif new_mode == "text":
                            mode = "text"
                            print("Text mode active")
                        else:
                            print("Invalid mode. Use: /mode text or /mode voice")
                        continue
                    
                    if text.startswith('/voice '):
                        parts = text.split(maxsplit=1)
                        if len(parts) == 2:
                            self.voice_id = parts[1]
                            print(f"Voice ID set to: {self.voice_id}")
                        continue
                    
                    if text.startswith('/format'):
                        parts = text.split()
                        if len(parts) >= 3:
                            try:
                                encoding = parts[1]
                                sample_rate = int(parts[2])
                                if encoding in ["pcm_f32le", "pcm_s16le"] and 8000 <= sample_rate <= 48000:
                                    self.set_output_format(encoding=encoding, sample_rate=sample_rate)
                                else:
                                    print("Invalid format. Encoding: pcm_f32le or pcm_s16le, Sample rate: 8000-48000")
                            except ValueError:
                                print("Invalid sample rate.")
                        continue
                    
                    if text == '/help':
                        print("\nCommands:")
                        print("  /mode <text|voice>: Switch input mode")
                        print("  /voice <id>: Change voice ID")
                        print("  /format <encoding> <sample_rate>: Change output format")
                        print("  /help: Show this help")
                        continue
                    
                    start_time = datetime.now()
                    generation_time = self.speak(text)
                    
                    if generation_time is not None:
                        total_time = (datetime.now() - start_time).total_seconds()
                        print(f"✓ Responded in {total_time:.2f} seconds (generation: {generation_time:.2f}s)\n")
                    else:
                        print("✗ Failed to generate speech\n")
                        
        except KeyboardInterrupt:
            print("\n\nExiting interactive session...")
            self.stop()
            print("Goodbye!")
        except Exception as e:
            print(f"\nError: {e}")
            print("Exiting...")
    
    def set_output_format(self, container: str = "raw", encoding: str = "pcm_f32le", sample_rate: int = 44100, bit_rate: Optional[int] = None) -> None:
        """
        Configure output format settings.
        
        Args:
            container (str): Output container format (e.g., "raw", "mp3")
            encoding (str): Audio encoding format (e.g., "pcm_f32le", "pcm_s16le")
            sample_rate (int): Sample rate in Hz (e.g., 44100)
            bit_rate (Optional[int]): Bit rate for compressed formats (e.g., 128000)

        """
        self.output_format = {
            "container": container,
            "encoding": encoding,
            "sample_rate": sample_rate
        }
        
        if container == "mp3" and bit_rate:
            self.output_format["bit_rate"] = bit_rate
        
        print(f"Output format updated: {self.output_format}")


if __name__ == "__main__":
    API_KEY = os.environ.get("ASYNC_API_KEY")
    BASE_URL = "https://api.async.ai"
    VOICE_ID = os.environ.get("ASYNC_VOICE_ID")
    VOSK_MODEL_PATH = os.environ.get("VOSK_MODEL_PATH")  
    
    if not API_KEY:
        print("Error: API key not found!")
        print("Please ensure you have a .env file with ASYNC_API_KEY=sk_xxxxx")
        print("Or set the ASYNC_API_KEY environment variable")
        exit(1)
    
    synth = HozieVoiceSynthesizer(
        api_key=API_KEY, 
        base_url=BASE_URL, 
        voice_id=VOICE_ID,
        vosk_model_path=VOSK_MODEL_PATH
    )
    
    if synth.is_initialized:
        synth.interactive_session()
    else:
        print("Failed to initialize TTS. Please check your API key and connection.")