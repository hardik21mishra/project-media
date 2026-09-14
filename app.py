from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse

import shutil 
import os
from transcribe import transcribe_audio
from pathlib import Path
from uuid import uuid4
from drive_upload import upload_to_drive
from video_audio import convert_video_to_audio

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
DOWNLOAD_DIR = Path(BASE_DIR) / "downloads"
OUTPUT_DIR = Path(BASE_DIR) / "output"
os.makedirs(UPLOADS_DIR, exist_ok=True)
DOWNLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def write_transcript(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


ALLOWED_OUTPUT_FORMATS = {"mp3", "wav", "m4a", "flac", "aac", "ogg", "opus"}

@app.post("/convert/upload")
async def convert_uploaded_video(
    file: UploadFile = File(...),
    output_format: str = Form("mp3"),
):
    output_format = output_format.lower().strip()

    if output_format not in ALLOWED_OUTPUT_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported output format. Choose one of: {sorted(ALLOWED_OUTPUT_FORMATS)}",
        )

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename was provided.")

    input_suffix = Path(file.filename).suffix or ".bin"
    input_path = DOWNLOAD_DIR / f"{uuid4().hex}{input_suffix}"

    try:
        with input_path.open("wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                buffer.write(chunk)

        output_path = convert_video_to_audio(
            input_path=input_path,
            output_dir=OUTPUT_DIR,
            output_format=output_format,
        )

        return FileResponse(
            path=output_path,
            filename=output_path.name,
            media_type="application/octet-stream",
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    finally:
        if input_path.exists():
            input_path.unlink(missing_ok=True)


@app.post("/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...)):
    
    save_path = os.path.join(UPLOADS_DIR, os.path.basename(file.filename))
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    transcript_path = save_path.rsplit(".", 1)[0] + ".txt"
    drive_error = None
    drive_file_ids = []

    try:
        transcript = transcribe_audio(save_path)
        write_transcript(transcript_path, transcript)

        try:
            drive_file_ids.append(upload_to_drive(save_path))
            drive_file_ids.append(upload_to_drive(transcript_path))
        except Exception as error:
            drive_error = str(error)
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)
        if os.path.exists(transcript_path):
            os.remove(transcript_path)

    response = {"transcript": transcript}
    response["drive_upload_ids"] = drive_file_ids
    if drive_error:
        response["drive_upload_error"] = drive_error
    return response