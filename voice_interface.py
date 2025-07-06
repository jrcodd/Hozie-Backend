import sounddevice as sd
import numpy as np
from datetime import datetime
import requests
import os
import json
import queue
import vosk
import threading
import time
from typing import Optional, Tuple, Any
from dotenv import load_dotenv
from Session import Session
from concurrent.futures import ThreadPoolExecutor  

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
        
        # Audio chunk queue for parallel processing
        self.audio_chunk_queue = queue.Queue()
        self.is_playing = False
        
        # Connection session for reuse
        self.requests_session = requests.Session()
        self.requests_session.headers.update({
            "X-Api-Key": self.api_key,
            "Content-Type": "application/json"
        })
        
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
    
    def _split_text_into_chunks(self, text: str, max_chunk_size: int = 200) -> list:
        """
        Split text into smaller chunks for faster processing.
        
        Args:
            text (str): Text to split
            max_chunk_size (int): Maximum characters per chunk
            
        Returns:
            list: List of text chunks
        """
        # First, split by sentences
        sentences = []
        current_sentence = ""
        
        for char in text:
            current_sentence += char
            if char in '.!?':
                sentences.append(current_sentence.strip())
                current_sentence = ""
        
        # Add any remaining text
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        # Group sentences into chunks
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed max_chunk_size, start a new chunk
            if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _generate_audio_chunk(self, chunk: str, chunk_index: int, voice_id: Optional[str] = None, retry_count: int = 0) -> Optional[Tuple[np.ndarray, int, int]]:
        """
        Generate audio for a single chunk (used in parallel processing).
        
        Args:
            chunk (str): Text chunk to convert
            chunk_index (int): Index of the chunk for ordering
            voice_id (str): Optional voice ID to use
            retry_count (int): Number of retry attempts
            
        Returns:
            Optional[Tuple[np.ndarray, int, int]]: (audio_array, sample_rate, chunk_index) or None if failed
        """
        
        voice_id = voice_id or self.voice_id
        
        try:
            data = {
                "model_id": "asyncflow_v2.0",
                "transcript": chunk,
                "voice": {
                    "mode": "id",
                    "id": voice_id
                },
                "output_format": self.output_format
            }
            
            # Add small delay to avoid overwhelming the API
            if chunk_index > 0:
                time.sleep(0.1 * chunk_index)
            
            response = self.requests_session.post(
                self.endpoint,
                json=data,
                stream=True,
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"API Error for chunk {chunk_index}: {response.status_code}")
                # Retry once for connection errors
                if retry_count == 0 and response.status_code >= 500:
                    time.sleep(1)
                    return self._generate_audio_chunk(chunk, chunk_index, voice_id, retry_count + 1)
                return None
            
            audio_chunks = []
            for chunk_data in response.iter_content(chunk_size=4096):
                if chunk_data:
                    if b"--ERROR:QUOTA_EXCEEDED--" in chunk_data:
                        print(f"Error: Quota exceeded for chunk {chunk_index}")
                        return None
                    audio_chunks.append(chunk_data)
            
            audio_data = b''.join(audio_chunks)
            
            if not audio_data:
                print(f"No audio data received for chunk {chunk_index}")
                return None
            
            audio_array, sample_rate = self._decode_audio_stream(
                audio_data, 
                self.output_format["encoding"],
                self.output_format["sample_rate"]
            )
            
            if audio_array.ndim == 1:
                audio_array = audio_array.reshape(-1, 1)
            
            return (audio_array, sample_rate, chunk_index)
            
        except Exception as e:
            print(f"Error generating audio for chunk {chunk_index}: {e}")
            # Retry once for connection errors
            if retry_count == 0:
                time.sleep(1)
                return self._generate_audio_chunk(chunk, chunk_index, voice_id, retry_count + 1)
            return None
    
    def _audio_playback_worker(self, total_chunks) -> None:
        """
        Worker thread that plays audio chunks in order.

        Args:
            total_chunks (int): Total number of audio chunks to play.
        """
        
        audio_buffer = {}
        next_chunk_index = 0
        chunks_played = 0
        
        while chunks_played < total_chunks:
            try:
                # Get next audio chunk from queue
                audio_data = self.audio_chunk_queue.get(timeout=10.0)
                
                if audio_data is None: 
                    if chunks_played >= total_chunks:
                        break
                    else:
                        continue  
                
                audio_array, sample_rate, chunk_index = audio_data
                audio_buffer[chunk_index] = (audio_array, sample_rate)
                
                while next_chunk_index in audio_buffer:
                    audio_array, sample_rate = audio_buffer.pop(next_chunk_index)
                    print(f"Playing chunk {next_chunk_index + 1}/{total_chunks}")
                    sd.play(audio_array, sample_rate)
                    sd.wait()
                    next_chunk_index += 1
                    chunks_played += 1
                
            except queue.Empty:
                print(f"Timeout waiting for chunk {next_chunk_index + 1}/{total_chunks}")
                continue
            except Exception as e:
                print(f"Error in audio playback worker: {e}")
                break
        
        print(f"✓ Finished playing all {chunks_played} chunks")
    
    def speak_chunks(self, text: str, voice_id: Optional[str] = None, speed: float = 1.0, max_chunk_size: int = 200) -> Optional[float]:
        """
        Convert text to speech in chunks with parallel processing for faster response time.
        
        Args:
            text (str): Text to convert to speech
            voice_id (str): Optional voice ID to use for synthesis
            speed (float): Speed factor for speech synthesis (default is 1.0)
            max_chunk_size (int): Maximum characters per chunk
        
        Returns:
            Optional[float]: Total time taken to generate all speech chunks, or None if failed
        """
        
        if not self.is_initialized:
            print("TTS not initialized properly")
            return None
        
        if not text or not isinstance(text, str):
            print("Invalid text input")
            return None
        
        chunks = self._split_text_into_chunks(text, max_chunk_size)
        
        if not chunks:
            return None
        
        start_time = datetime.now()
        
        while not self.audio_chunk_queue.empty():
            try:
                self.audio_chunk_queue.get_nowait()
            except queue.Empty:
                break
        
        self.is_playing = True
        playback_thread = threading.Thread(target=self._audio_playback_worker, args=(len(chunks),))
        playback_thread.daemon = True
        playback_thread.start()
        
        def generate_and_queue(chunk, index):
            audio_data = self._generate_audio_chunk(chunk, index, voice_id)
            if audio_data:
                self.audio_chunk_queue.put(audio_data)
        
        # Limit to 5 concurrent requests since that is what async.ai allows for on the $1/hr plan
        max_workers = min(5, len(chunks))  
        print(f"Generating {len(chunks)} chunks with {max_workers} concurrent workers...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, chunk in enumerate(chunks):
                print(f"Queuing chunk {i+1}/{len(chunks)}: {chunk[:50]}...")
                future = executor.submit(generate_and_queue, chunk, i)
                futures.append(future)
            
            # Wait for all generation tasks to complete
            for future in futures:
                future.result()
        
        self.audio_chunk_queue.put(None)
        self.is_playing = False
        
        playback_thread.join()
        
        total_time = (datetime.now() - start_time).total_seconds()
        return total_time
    
    def audio_callback(self, indata: np.ndarray, frames: int, time: Any, status: sd.CallbackFlags) -> None:
        """
        Callback for audio recording.

        Args:
            indata (np.ndarray): Input audio data
            frames (int): Number of frames in the input data
            time (Any): Time information (not used)
            status (sd.CallbackFlags): Status flags for the callback
        """
        
        if status:
            print(f"Audio callback status: {status}")
        self.audio_queue.put(bytes(indata))
    
    def listen_for_speech(self, timeout: float = 30.0) -> Optional[str]:
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
        max_silence = 3.0  # Stop after 3 seconds of silence
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
                
                if any(word in user_input.lower() for word in ['goodbye', 'bye', 'exit', 'quit', 'q']):
                    print("\nGoodbye!")
                    self.speak("Goodbye!")
                    break

                print(f"\nProcessing: '{user_input}'")

                start_time = datetime.now()
                try:
                    response = self.session.answer(user_input)
                    brain_time = (datetime.now() - start_time).total_seconds()
                    print(f"Brain responded in {brain_time:.2f}s")
                except Exception as e:
                    print(f"Brain error: {e}")
                    response = "I'm sorry, I encountered an error processing that."
                
                # replace these words that the tts model has a hard time saying
                response = response.replace("haha", "").replace("Aight", "ight").replace("*", "").replace("ya", "yeuh")
                print(f"\nSpeaking: {response}")
                generation_time = self.speak_chunks(response)
                
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