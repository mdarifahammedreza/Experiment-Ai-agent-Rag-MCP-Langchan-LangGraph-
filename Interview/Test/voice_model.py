import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import whisper
import av
import torch
import numpy as np
import threading
import time
from collections import deque
import os
from datetime import datetime
from gtts import gTTS
import tempfile
import io

# Global variables for thread safety
is_processing = False
transcript_text = ""
auto_save_enabled = False

def main():
    global is_processing, transcript_text
    print("=== Starting Voice Model ===")
    
    # Load Whisper model (tiny is fastest; use "base", "small", etc. for better accuracy)
    st.write("=== CUDA Diagnostics ===")
    st.write(f"PyTorch version: {torch.__version__}")
    st.write(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        st.write(f"CUDA version: {torch.version.cuda}")
        st.write(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            st.write(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        st.error("CUDA is not available. Possible reasons:")
        st.write("1. PyTorch was installed without CUDA support")
        st.write("2. NVIDIA drivers are not installed")
        st.write("3. No NVIDIA GPU detected")
        st.write("4. CUDA toolkit is not installed")
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    st.write(f"Using device: {DEVICE}")
    print(f"Using device: {DEVICE}")
    
    with st.spinner("Loading Whisper model..."):
        model = whisper.load_model("tiny", device=DEVICE)
        print("Whisper model loaded successfully")
    
    st.title("🎤 Live Voice to Text & Text to Speech")
    
    st.markdown("Speak into your mic and see real-time transcription below:")
    
    # Initialize session state
    if "transcript" not in st.session_state:
        st.session_state.transcript = ""

    # Very simple audio processor
    class SimpleAudioProcessor:
        def __init__(self, model, device):
            self.model = model
            self.device = device
            self.audio_buffer = []
            self.frame_count = 0
            print("Audio processor initialized")

        def recv(self, frame):
            global is_processing, transcript_text
            
            # Convert audio frame to numpy array
            audio = frame.to_ndarray()
            
            # Convert to mono if stereo
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            
            # Add to buffer
            self.audio_buffer.extend(audio)
            self.frame_count += 1
            
            # Transcribe every 100 frames (about 3-4 seconds of audio)
            if self.frame_count >= 100 and len(self.audio_buffer) > 0:
                try:
                    # Get audio data
                    audio_data = np.array(self.audio_buffer)
                    
                    # Normalize audio
                    audio_data = audio_data.astype(np.float32) / 32768.0
                    
                    # Only transcribe if there's enough audio and we're not already processing
                    if len(audio_data) > 8000 and not is_processing:
                        print("🔄 Transcribing audio...")
                        # Start transcription in background
                        threading.Thread(target=self._transcribe_audio, args=(audio_data.copy(),)).start()
                    
                    # Clear buffer after processing
                    self.audio_buffer = []
                    self.frame_count = 0
                    
                except Exception as e:
                    print(f"Audio processing error: {e}")
            
            return frame

        def _transcribe_audio(self, audio_data):
            global is_processing, transcript_text, auto_save_enabled
            try:
                is_processing = True
                
                # Transcribe using Whisper
                result = self.model.transcribe(
                    audio_data, 
                    language="en", 
                    fp16=False,
                    task="transcribe"
                )
                
                # Update transcript if we got text
                if result["text"].strip():
                    transcript_text += result["text"].strip() + " "
                    print(f"📝 Transcribed: {result['text'].strip()}")
                    
                    # Auto-save if enabled
                    if auto_save_enabled:
                        try:
                            # Create transcripts directory if it doesn't exist
                            os.makedirs("transcripts", exist_ok=True)
                            
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = os.path.join("transcripts", f"transcript_{timestamp}.txt")
                            
                            with open(filename, "w", encoding="utf-8") as f:
                                f.write(f"Voice Transcription - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                                f.write("=" * 50 + "\n\n")
                                f.write(transcript_text)
                            
                            print(f"💾 Auto-saved to {filename}")
                        except Exception as save_error:
                            print(f"Auto-save error: {save_error}")
                    
            except Exception as e:
                print(f"Transcription error: {e}")
            finally:
                is_processing = False

    # Create audio processor
    audio_processor = SimpleAudioProcessor(model, DEVICE)

    # WebRTC streamer
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎙️ Microphone")
        try:
            print("Setting up WebRTC streamer...")
            webrtc_streamer(
                key="live-whisper",
                mode=WebRtcMode.SENDONLY,
                media_stream_constraints={"audio": True, "video": False},
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                audio_frame_callback=audio_processor.recv,
                async_processing=True,
            )
            print("WebRTC streamer created")
        except Exception as e:
            print(f"WebRTC Error: {e}")
            st.error(f"WebRTC Error: {e}")
            st.info("Please make sure you're running this with: streamlit run voice_model.py")

    with col2:
        st.subheader("📊 Status")
        if is_processing:
            st.info("🔄 Processing audio...")
        else:
            st.success("✅ Ready")
        
        # File operations
        st.subheader("💾 Save Transcript")
        
        # Auto-save toggle
        global auto_save_enabled, transcript_text
        auto_save_enabled = st.checkbox("🔄 Auto-save transcript", value=auto_save_enabled)
        
        # Save to file button
        if st.button("💾 Save to File"):
            if transcript_text.strip():
                try:
                    # Create transcripts directory if it doesn't exist
                    os.makedirs("transcripts", exist_ok=True)
                    
                    # Create filename with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join("transcripts", f"transcript_{timestamp}.txt")
                    
                    # Save to file
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(f"Voice Transcription - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write("=" * 50 + "\n\n")
                        f.write(transcript_text)
                    
                    st.success(f"✅ Transcript saved to {filename}")
                    print(f"Transcript saved to {filename}")
                except Exception as e:
                    st.error(f"Error saving file: {e}")
                    print(f"Error saving file: {e}")
            else:
                st.warning("No transcript to save")
        
        # Download button
        if transcript_text.strip():
            st.download_button(
                label="📥 Download Transcript",
                data=transcript_text,
                file_name=f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        # Clear transcript button
        if st.button("🗑️ Clear Transcript"):
            transcript_text = ""
            st.session_state.transcript = ""
            print("Transcript cleared")

    # Update session state from global variable
    st.session_state.transcript = transcript_text

    # Real-time transcript display
    st.subheader("📝 Live Transcription")
    
    # Display the transcript
    if st.session_state.transcript:
        st.markdown(f"**Transcript:** {st.session_state.transcript}")
        print(f"Displaying transcript: {st.session_state.transcript}")
    else:
        st.info("Start speaking to see transcription here...")
    
    # Text to Speech Section
    st.markdown("---")
    st.subheader("🔊 Text to Speech")
    
    # Text input for TTS
    tts_text = st.text_area("Enter text to convert to speech:", height=100)
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Language selection
        language = st.selectbox("Language", ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"], index=0)
        
        # Voice speed
        slow_speech = st.checkbox("Slow Speech", value=False)
    
    with col4:
        # TTS buttons
        if st.button("🔊 Speak Text"):
            if tts_text.strip():
                try:
                    with st.spinner("Generating speech..."):
                        # Create TTS object
                        tts = gTTS(text=tts_text, lang=language, slow=slow_speech)
                        
                        # Create temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                            temp_filename = fp.name
                        
                        # Save audio to file
                        tts.save(temp_filename)
                        
                        # Play the audio
                        st.audio(temp_filename, format="audio/mp3")
                        
                        # Clean up
                        os.unlink(temp_filename)
                    
                    st.success("✅ Text converted to speech!")
                    print(f"🔊 Generated speech for: {tts_text[:50]}...")
                    
                except Exception as e:
                    st.error(f"TTS Error: {e}")
                    print(f"TTS Error: {e}")
            else:
                st.warning("Please enter some text to speak")
        
        # Speak transcript button
        if st.button("🔊 Speak Transcript"):
            if transcript_text.strip():
                try:
                    with st.spinner("Generating speech..."):
                        # Create TTS object
                        tts = gTTS(text=transcript_text, lang=language, slow=slow_speech)
                        
                        # Create temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                            temp_filename = fp.name
                        
                        # Save audio to file
                        tts.save(temp_filename)
                        
                        # Play the audio
                        st.audio(temp_filename, format="audio/mp3")
                        
                        # Clean up
                        os.unlink(temp_filename)
                    
                    st.success("✅ Transcript converted to speech!")
                    print(f"🔊 Generated speech for transcript: {transcript_text[:50]}...")
                    
                except Exception as e:
                    st.error(f"TTS Error: {e}")
                    print(f"TTS Error: {e}")
            else:
                st.warning("No transcript to speak")
    
    # Add some spacing
    st.write("")
    st.write("")
    
    # Instructions
    st.markdown("---")
    st.markdown("### Instructions:")
    st.markdown("1. Click 'START' on the microphone widget above")
    st.markdown("2. Allow microphone access when prompted")
    st.markdown("3. Start speaking - transcription will appear below")
    st.markdown("4. Use Text-to-Speech to hear your text or transcript")
    st.markdown("5. Save or download your transcript")
    st.markdown("6. Click 'STOP' when done")
    
    # Debug info
    with st.expander("🔧 Debug Information"):
        st.write(f"Audio buffer size: {len(audio_processor.audio_buffer)}")
        st.write(f"Frame count: {audio_processor.frame_count}")
        st.write(f"Is processing: {is_processing}")
        st.write(f"Transcript length: {len(st.session_state.transcript)}")
        st.write(f"Current transcript: {st.session_state.transcript}")
        
        # Show saved files
        if os.path.exists("transcripts"):
            files = os.listdir("transcripts")
            st.write(f"Saved transcript files: {files}")

if __name__ == "__main__":
    main()
