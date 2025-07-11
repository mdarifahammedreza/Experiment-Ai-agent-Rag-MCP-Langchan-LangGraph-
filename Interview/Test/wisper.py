import whisper 

model = whisper.load_model('base')
result = model.transcribe('Recording.mp3',fp16=False)
result['text']