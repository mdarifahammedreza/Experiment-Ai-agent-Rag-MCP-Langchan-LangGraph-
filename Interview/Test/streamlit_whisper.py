import streamlit as st
import whisper
import tempfile
import os

def main():
    st.title("Audio Transcription with Whisper")
    st.write("Upload an audio file to transcribe it using OpenAI's Whisper model.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file", 
        type=['mp3', 'wav', 'm4a', 'ogg', 'flac']
    )
    
    if uploaded_file is not None:
        # Display file info
        st.write(f"**File uploaded:** {uploaded_file.name}")
        st.write(f"**File size:** {uploaded_file.size} bytes")
        
        # Load Whisper model
        with st.spinner("Loading Whisper model..."):
            model = whisper.load_model("base")  # Using base model for faster processing
        
        # Process the audio file
        with st.spinner("Transcribing audio..."):
            try:
                # Save uploaded file to temporary location with proper extension
                file_extension = uploaded_file.name.split('.')[-1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Verify file exists and is readable
                if not os.path.exists(tmp_file_path):
                    st.error(f"Temporary file not created: {tmp_file_path}")
                    return
                
                st.write(f"Processing file: {tmp_file_path}")
                
                # Use whisper's transcribe function directly for better compatibility
                result = model.transcribe(tmp_file_path)
                
                # Get detected language from result
                detected_language = result.get("language", "unknown")
                language_prob = result.get("language_prob", 0.0)
                
                # Display results
                st.success("Transcription completed!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Detected Language:** {detected_language}")
                with col2:
                    st.write(f"**Confidence:** {language_prob:.2%}")
                
                st.subheader("Transcribed Text:")
                transcribed_text = result.get("text", "")
                st.write(transcribed_text)
                
                # Add copy button
                if st.button("Copy to Clipboard"):
                    st.write("Text copied to clipboard!")
                    st.code(transcribed_text)
                
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
    
    else:
        st.info("Please upload an audio file to get started.")

if __name__ == "__main__":
    main() 