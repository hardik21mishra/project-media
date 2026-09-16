import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from groq import Groq

from drive_upload import upload_to_drive
from media_download import MediaDownloadError, download_media
from transcribe import transcribe_audio
from video_audio import convert_video_to_audio
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

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

jobs = {}
active_job_id = None
conversations = {}


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

def job_result(
    audio_path: Path,
    transcript_path: Path,
    audio_url: str,
    transcript_url: str,
):
    return {
        "message": "Files ready",
        "audio_url": audio_url,
        "transcript_url": transcript_url,
        "audio_filename": audio_path.name,
        "transcript_filename": transcript_path.name,
    }

def transcribe_and_upload(job_id: str, audio_path: Path):
    transcript_path = audio_path.with_suffix(".txt")
    transcript = transcribe_audio(str(audio_path))
    write_transcript(transcript_path, transcript)
    audio_url = upload_to_drive(str(audio_path))
    transcript_url = upload_to_drive(str(transcript_path))
    return job_result(audio_path, transcript_path, audio_url, transcript_url)


def finish_job(job_id: str, audio_path: Path):
    result = transcribe_and_upload(job_id, audio_path)
    jobs[job_id] = {
        "status": "done",
        "result": result,
    }


def run_transcription_job(job_id: str, audio_path: Path):
    global active_job_id
    try:
        finish_job(job_id, audio_path)
    except Exception as exc:
        jobs[job_id] = {"status": "failed", "error": str(exc)}
    finally:
        active_job_id = None


def run_uploaded_conversion_job(
    job_id: str,
    input_path: Path,
    output_format: str,
):
    global active_job_id
    try:
        audio_path = convert_video_to_audio(
            input_path=input_path,
            output_dir=input_path.parent,
            output_format=output_format,
        )
        input_path.unlink(missing_ok=True)
        finish_job(job_id, audio_path)
    except Exception as exc:
        jobs[job_id] = {"status": "failed", "error": str(exc)}
    finally:
        active_job_id = None


def run_url_conversion_job(
    job_id: str,
    url: str,
    job_dir: Path,
    output_format: str,
):
    global active_job_id
    try:
        input_path = download_media(url, job_dir)
        audio_path = convert_video_to_audio(
            input_path=input_path,
            output_dir=job_dir,
            output_format=output_format,
        )
        input_path.unlink(missing_ok=True)
        finish_job(job_id, audio_path)
    except Exception as exc:
        jobs[job_id] = {"status": "failed", "error": str(exc)}
    finally:
        active_job_id = None


def chat_reply(
    conversation_id: str,
    reply: str,
    job_id: str | None = None,
    status: str | None = None,
    result: dict | None = None,
    attachment: dict | None = None,
):
    response: dict[str, object] = {
        "conversation_id": conversation_id,
        "reply": reply,
    }
    if job_id:
        response["job_id"] = job_id
    if status:
        response["status"] = status
    if result:
        response["result"] = result
    if attachment:
        response["attachment"] = attachment
    return response


def ask_groq(conversation_id: str, message: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    conversation = conversations.setdefault(conversation_id, {"messages": []})
    messages = conversation.setdefault("messages", [])
    messages.append({"role": "user", "content": message})
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Voxera, a concise and friendly media assistant. "
                    "You can transcribe audio, convert videos to audio, and report job status. "
                    "Never claim a file was processed unless the backend says the job is done."
                ),
            },
            *messages[-10:],
        ],
        temperature=0.3,
        max_tokens=300,
    )
    reply = completion.choices[0].message.content or "How can I help with your media?"
    messages.append({"role": "assistant", "content": reply})
    return reply


@app.post("/chat")
async def chat_endpoint(
    background_tasks: BackgroundTasks,
    message: str = Form(""),
    conversation_id: str | None = Form(None),
    file: UploadFile | None = File(None),
    url: str | None = Form(None),
    output_format: str = Form("mp3"),
):
    global active_job_id

    conversation_id = conversation_id or uuid4().hex
    conversation = conversations.setdefault(conversation_id, {"messages": []})
    normalized_message = message.lower().strip()
    status_requested = any(
        phrase in normalized_message
        for phrase in ("status", "progress", "what happened", "still processing")
    )

    if status_requested and file is None and not url:
        job_id = conversation.get("last_job_id")
        job = jobs.get(job_id) if job_id else None
        if not job:
            return chat_reply(
                conversation_id,
                "I do not have a previous media job in this conversation yet.",
            )
        if job["status"] == "processing":
            return chat_reply(
                conversation_id,
                "Your latest media job is still processing.",
                job_id=job_id,
                status="processing",
            )
        if job["status"] == "failed":
            return chat_reply(
                conversation_id,
                f"Your latest media job failed: {job['error']}",
                job_id=job_id,
                status="failed",
            )
        return chat_reply(
            conversation_id,
            "Your latest media job is complete. You can download the files below.",
            job_id=job_id,
            status="done",
            result=job["result"],
        )

    wants_conversion = any(
        word in normalized_message
        for word in ("convert", "video", "mp4", "mkv", "mov", "avi", "webm")
    ) or bool(url)
    wants_transcription = any(
        word in normalized_message
        for word in ("transcribe", "transcription", "transcript", "audio to text")
    )

    if file is None and not url:
        if wants_conversion:
            return chat_reply(
                conversation_id,
                "Please upload a video or send a video URL. You can also choose an output format.",
            )
        if wants_transcription:
            return chat_reply(
                conversation_id,
                "Please upload the audio or video you want me to transcribe.",
            )
        try:
            reply = await run_in_threadpool(ask_groq, conversation_id, message)
        except Exception:
            reply = "I can transcribe audio, convert videos, and report media-job status. How can I help?"
        return chat_reply(conversation_id, reply)

    if active_job_id is not None:
        return chat_reply(
            conversation_id,
            "A previous media job is still processing. Please wait for it to finish before starting another one.",
            job_id=active_job_id,
            status="processing",
        )

    output_format = output_format.lower().strip()
    if output_format not in ALLOWED_OUTPUT_FORMATS:
        return chat_reply(
            conversation_id,
            f"Unsupported output format. Choose one of: {sorted(ALLOWED_OUTPUT_FORMATS)}.",
        )

    job_id = uuid4().hex
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir()
    jobs[job_id] = {"status": "processing"}
    conversation["last_job_id"] = job_id
    active_job_id = job_id

    if url:
        try:
            validated_url = validate_media_url(url)
        except HTTPException:
            jobs[job_id] = {"status": "failed", "error": "Invalid media URL."}
            active_job_id = None
            raise
        background_tasks.add_task(
            run_url_conversion_job, job_id, validated_url, job_dir, output_format
        )
        return chat_reply(
            conversation_id,
            "I started downloading, converting, and transcribing the video.",
            job_id=job_id,
            status="processing",
        )

    if file is None:
        raise HTTPException(
            status_code=400,
            detail="Provide an audio/video file or a media URL.",
        )

    filename = Path(file.filename or "uploaded_media").name
    input_path = job_dir / f"input{Path(filename).suffix or '.bin'}"
    try:
        await save_upload(file, input_path)
    except Exception as exc:
        jobs[job_id] = {"status": "failed", "error": str(exc)}
        active_job_id = None
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    attachment = {
        "filename": filename,
        "size_bytes": input_path.stat().st_size,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
    if wants_conversion:
        background_tasks.add_task(
            run_uploaded_conversion_job, job_id, input_path, output_format
        )
        reply = "Your video is processing. I will let you know when the converted audio and transcript are ready."
    else:
        background_tasks.add_task(run_transcription_job, job_id, input_path)
        reply = "Your transcription is processing. I will let you know when it is ready."

    return chat_reply(
        conversation_id,
        reply,
        job_id=job_id,
        status="processing",
        attachment=attachment,
    )


@app.post("/convert/url")
async def convert_video_url(
    background_tasks: BackgroundTasks,
    url: str = Form(...),
    output_format: str = Form("mp3"),
):
    global active_job_id

    output_format = output_format.lower().strip()
    if output_format not in ALLOWED_OUTPUT_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported output format. Choose one of: {sorted(ALLOWED_OUTPUT_FORMATS)}",
        )

    url = validate_media_url(url)
    if active_job_id is not None:
        raise HTTPException(
            status_code=409,
            detail="Another media job is already in process. Please wait for it to finish.",
        )

    job_id = uuid4().hex
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir()
    jobs[job_id] = {"status": "processing"}
    active_job_id = job_id
    background_tasks.add_task(
        run_url_conversion_job,
        job_id,
        url,
        job_dir,
        output_format,
    )
    return {
        "job_id": job_id,
        "status": "processing",
        "message": "Download, conversion, and transcription started.",
    }

@app.post("/convert/upload")
async def convert_uploaded_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    output_format: str = Form("mp3"),
):
    global active_job_id

    output_format = output_format.lower().strip()

    if output_format not in ALLOWED_OUTPUT_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported output format. Choose one of: {sorted(ALLOWED_OUTPUT_FORMATS)}",
        )

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename was provided.")

    if active_job_id is not None:
        raise HTTPException(
            status_code=409,
            detail="Another media job is already in process. Please wait for it to finish.",
        )

    job_id = uuid4().hex
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir()
    input_path = job_dir / f"input{Path(file.filename).suffix or '.bin'}"
    jobs[job_id] = {"status": "processing"}
    active_job_id = job_id

    try:
        await save_upload(file, input_path)
        background_tasks.add_task(
            run_uploaded_conversion_job,
            job_id,
            input_path,
            output_format,
        )
        return {
            "job_id": job_id,
            "status": "processing",
            "message": "Conversion and transcription started.",
        }

    except Exception as exc:
        jobs[job_id] = {"status": "failed", "error": str(exc)}
        active_job_id = None
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/transcribe")
async def transcribe_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    global active_job_id

    if active_job_id is not None:
        raise HTTPException(
            status_code=409,
            detail="A transcription is already in process. Please wait for it to finish.",
        )

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename was provided.")

    job_id = uuid4().hex
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir()
    audio_path = job_dir / (Path(file.filename).name or "audio")
    jobs[job_id] = {"status": "processing"}
    active_job_id = job_id
    try:
        await save_upload(file, audio_path)
        background_tasks.add_task(run_transcription_job, job_id, audio_path)
        return {
            "job_id": job_id,
            "status": "processing",
            "message": "Transcription started.",
        }
    except Exception as exc:
        jobs[job_id] = {"status": "failed", "error": str(exc)}
        active_job_id = None
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {"job_id": job_id, **job}

