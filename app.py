from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from html import escape
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from drive_upload import upload_to_drive
from media_download import MediaDownloadError, download_media
from transcribe import transcribe_audio
from video_audio import convert_video_to_audio
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def write_transcript(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


ALLOWED_OUTPUT_FORMATS = {"mp3", "wav", "m4a", "flac", "aac", "ogg", "opus"}


async def save_upload(file: UploadFile, path: Path) -> None:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename was provided.")

    with path.open("wb") as buffer:
        while chunk := await file.read(1024 * 1024):
            buffer.write(chunk)


def validate_media_url(url: str) -> str:
    parsed_url = urlsplit(url.strip())
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise HTTPException(
            status_code=400,
            detail="Provide a valid HTTP(S) video URL.",
        )
    return parsed_url.geturl()

def job_response(
    audio_path: Path,
    transcript_path: Path,
    audio_url: str,
    transcript_url: str,
):
    return JSONResponse(
        content={
            "message": "Files ready",
            "audio_url": audio_url,
            "transcript_url": transcript_url,
            "audio_filename": audio_path.name,
            "transcript_filename": transcript_path.name,
        }
    )

def transcribe_and_upload(job_id: str, audio_path: Path):
    transcript_path = audio_path.with_suffix(".txt")
    transcript = transcribe_audio(str(audio_path))
    write_transcript(transcript_path, transcript)
    audio_url = upload_to_drive(str(audio_path))
    transcript_url = upload_to_drive(str(transcript_path))
    return job_response(audio_path, transcript_path, audio_url, transcript_url)


@app.post("/convert/url")
async def convert_video_url(
    url: str = Form(...),
    output_format: str = Form("mp3"),
):
    """Download one video URL, convert its audio, and return audio/transcript links."""
    output_format = output_format.lower().strip()
    if output_format not in ALLOWED_OUTPUT_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported output format. Choose one of: {sorted(ALLOWED_OUTPUT_FORMATS)}",
        )

    url = validate_media_url(url)
    job_id = uuid4().hex
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir()

    try:
        input_path = download_media(url, job_dir)
        audio_path = convert_video_to_audio(
            input_path=input_path,
            output_dir=job_dir,
            output_format=output_format,
        )
        input_path.unlink(missing_ok=True)
        return transcribe_and_upload(job_id, audio_path)
    except MediaDownloadError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/convert/upload", response_class=HTMLResponse)
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

    job_id = uuid4().hex
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir()
    input_path = job_dir / f"input{Path(file.filename).suffix or '.bin'}"

    try:
        await save_upload(file, input_path)
        audio_path = convert_video_to_audio(
            input_path=input_path,
            output_dir=job_dir,
            output_format=output_format,
        )

        input_path.unlink(missing_ok=True)
        return transcribe_and_upload(job_id, audio_path)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/transcribe", response_class=HTMLResponse)
async def transcribe_endpoint(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename was provided.")

    job_id = uuid4().hex
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir()
    audio_path = job_dir / (Path(file.filename).name or "audio")
    try:
        await save_upload(file, audio_path)
        return transcribe_and_upload(job_id, audio_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.get("/", response_class=HTMLResponse)
async def upload_page():
    return """<!doctype html>
<html><head><title>Audio transcription</title>
<style>body{font-family:system-ui;max-width:600px;margin:64px auto;padding:0 24px}form{margin:28px 0;padding:20px;border:1px solid #ddd;border-radius:8px}button{margin-top:12px;padding:10px 16px;background:#2563eb;color:white;border:0;border-radius:6px;font-weight:600}</style>
</head><body><h1>Audio transcription</h1>
<form action="/transcribe" method="post" enctype="multipart/form-data"><h2>Upload audio</h2><input type="file" name="file" required><br><button type="submit">Upload and transcribe</button></form>
<form action="/convert/upload" method="post" enctype="multipart/form-data"><h2>Upload video</h2><input type="file" name="file" required><br><label>Audio format <select name="output_format"><option>mp3</option><option>wav</option><option>m4a</option></select></label><br><button type="submit">Convert and transcribe</button></form>
<form action="/convert/url" method="post"><h2>Video URL</h2><input type="url" name="url" placeholder="https://..." required><br><label>Audio format <select name="output_format"><option>mp3</option><option>wav</option><option>m4a</option></select></label><br><button type="submit">Download, convert and transcribe</button></form>
</body></html>"""
