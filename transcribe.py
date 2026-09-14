import os
from dotenv import load_dotenv
import assemblyai as aai

load_dotenv()

aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

def transcribe_audio(audio_path):
    transcriber = aai.Transcriber()
    config = aai.TranscriptionConfig(
        speech_models=["universal-3-5-pro", "universal-2"],
        language_detection=True,
        punctuate=True,
        format_text=True,
        disfluencies=True,
        speaker_labels=os.getenv("TRANSCRIBE_SPEAKERS", "false").lower() == "true",
    )
    transcription = transcriber.transcribe(audio_path, config=config)

    if transcription.status != aai.TranscriptStatus.completed:
        raise Exception(f"Transcription failed: {transcription.error}")

    if not transcription.text or not transcription.text.strip():
        raise Exception("Transcription completed but no speech was detected in the audio.")

    return transcription.text
