from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4
import re
import logging
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, UploadFile, Form, HTTPException
from starlette.concurrency import run_in_threadpool
from chat_service import IntentDecision, chat_reply, classify_request, record_message
from file_service import conversations, get_files
from job_service import (
    get_active_job_id,
    has_active_job,
    jobs,
    claim_job,
    release_job,
    run_transcription_job,
    run_uploaded_conversion_job,
    run_url_conversion_job,
    submit_job,
)
from fastapi.middleware.cors import CORSMiddleware
load_dotenv()
logger = logging.getLogger("uvicorn.error")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_OUTPUT_FORMATS = {"mp3", "wav", "m4a", "flac", "aac", "ogg", "opus"}

UPLOAD_INTENT_PATTERN = re.compile(
    r"\b(?:upload|attach)\b"
)
MEDIA_INPUT_PATTERN = re.compile(
    r"\b(?:audio|video|media|sound|recording)(?:\s+(?:file|clip|recording|track))?\b"
)
PROCESSING_REQUEST_PATTERN = re.compile(
    r"\b(?:process|processed|processing|handle|work\s+on|analyze|analyse|convert|transcribe)\b"
)
TRANSCRIPTION_INTENT_PATTERN = re.compile(
    r"\b(?:transcribe|transcription|transcript|speech\s+to\s+text|audio\s+to\s+text)\b"
)
CONVERSION_INTENT_PATTERN = re.compile(
    r"\b(?:convert|conversion|extract\s+(?:the\s+)?(?:audio|sound))\b"
)
STATUS_INTENT_PATTERN = re.compile(
    r"\b(?:status|progress|what\s+happened|still\s+processing|in\s+process)\b"
    r"|^(?:is\s+it\s+)?(?:finished|done|complete|ready)[?.!]*$"
)
TARGET_FORMAT_PATTERN = re.compile(
    r"\b(?:to|as|in|use|format(?:\s+is|\s+to)?)[\s:=]+"
    r"(mp3|wav|m4a|flac|aac|ogg|opus|mp4|wma|aiff)\b", re.I
)


def detect_intents(message: str) -> set[str]:
    """Conservative fallback when the semantic classifier is unavailable."""
    normalized_message = " ".join(message.casefold().split())
    intents = set()
    has_media_input = bool(MEDIA_INPUT_PATTERN.search(normalized_message))
    has_processing_request = bool(PROCESSING_REQUEST_PATTERN.search(normalized_message))
    if re.search(r"\b(?:don't|do not|never|cancel)\b", normalized_message):
        return intents
    if re.match(r"^(?:what (?:is|are)|explain|define)\b", normalized_message):
        return intents
    if UPLOAD_INTENT_PATTERN.search(normalized_message) or (
        has_media_input and (has_processing_request or re.search(
            r"\b(?:choose|select|send|add|provide|share)\b", normalized_message
        ))
    ):
        intents.add("upload")
    if TRANSCRIPTION_INTENT_PATTERN.search(normalized_message):
        intents.add("transcription")
    if CONVERSION_INTENT_PATTERN.search(normalized_message):
        intents.add("conversion")
    if STATUS_INTENT_PATTERN.search(normalized_message):
        intents.add("status")
    return intents


def sync_job_context(conversation_id: str, conversation: dict) -> None:
    job = jobs.get(conversation.get("last_job_id"))
    if not job:
        return
    status = job.get("status")
    if status != conversation.get("job_status") and status in {"done", "failed"}:
        event = "The latest media job completed; its files are ready." if status == "done" else "The latest media job failed."
        record_message(conversation_id, "assistant", event)
    conversation["job_status"] = status


async def interpret_chat(message: str, conversation: dict, has_media: bool) -> IntentDecision:
    normalized = " ".join(message.casefold().split()).strip(".!?")
    pending = conversation.get("pending_task")
    if normalized in {"cancel", "never mind", "nevermind", "forget it", "cancel upload"}:
        return IntentDecision("cancel")
    if not normalized and has_media:
        return IntentDecision(pending or "transcription")
    if normalized in {"upload", "upload file", "upload a file", "attach a file"}:
        return IntentDecision("upload")
    if pending and normalized in {"yes", "yes please", "okay", "ok", "go ahead", "ready", "i'm ready", "i am ready"}:
        return IntentDecision(pending)
    format_match = TARGET_FORMAT_PATTERN.fullmatch(normalized)
    if pending and format_match:
        return IntentDecision(pending if pending != "upload" else "conversion", format_match.group(1))
    if STATUS_INTENT_PATTERN.fullmatch(normalized):
        return IntentDecision("status")
    try:
        context = {**conversation, "media_attached_this_turn": has_media}
        decision = await run_in_threadpool(classify_request, message, context)
        logger.info("chat_intent source=groq intent=%s", decision.intent)
        return decision
    except Exception as exc:
        # Do not log prompts, file contents, credentials, or provider response bodies.
        logger.warning("chat_intent source=fallback error_type=%s", type(exc).__name__)
        intents = detect_intents(message)
        intent = next((candidate for candidate in ("transcription", "conversion", "upload", "status") if candidate in intents), None)
        if intent is None:
            if has_media:
                intent = pending or "transcription"
            elif MEDIA_INPUT_PATTERN.search(normalized):
                intent = "clarification"
            else:
                intent = "general"
        target = TARGET_FORMAT_PATTERN.search(message)
        return IntentDecision(intent, target.group(1).lower() if target else None)

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

def get_status_job(conversation: dict):
    job_id = conversation.get("last_job_id")
    return job_id, jobs.get(job_id) if job_id else None

@app.post("/chat")
async def chat_endpoint(
    background_tasks: BackgroundTasks,
    message: str = Form(""),
    conversation_id: str | None = Form(None),
    file: UploadFile | None = File(None),
    url: str | None = Form(None),
    output_format: str | None = Form(None),
):
    conversation_id = conversation_id or uuid4().hex
    conversation = conversations.setdefault(conversation_id, {"messages": []})
    sync_job_context(conversation_id, conversation)
    decision = await interpret_chat(message, conversation, file is not None or bool(url))
    intent = decision.intent
    logger.info("chat_intent conversation=%s intent=%s has_media=%s", conversation_id, intent, file is not None or bool(url))
    user_content = message
    if file is not None:
        user_content += f"\n[Attached media: {Path(file.filename or 'media').name}]"
    if url and url not in message:
        user_content += f"\n[Media URL: {url}]"
    record_message(conversation_id, "user", user_content.strip())

    if intent == "cancel":
        for key in ("pending_task", "pending_output_format", "awaiting_media"):
            conversation.pop(key, None)
        return chat_reply(conversation_id, "The pending upload request is cleared. Any job already running will continue.")

    if intent == "clarification":
        return chat_reply(conversation_id, decision.reply or "Would you like a written transcript or to convert the media to another format?")

    # A picker submission has no text: continue the task chosen before upload.
    if (file is not None or url) and intent in {"general", "upload", "status"}:
        intent = conversation.get("pending_task") or "transcription"
    wants_upload = intent == "upload"
    status_requested = intent == "status"

    if status_requested and file is None and not url:
        job_id, job = get_status_job(conversation)
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

    wants_conversion = intent == "conversion" or bool(url)
    wants_transcription = intent == "transcription"

    # Explicit choices override the saved preference; an omitted field resumes it.
    if decision.output_format is not None:
        output_format = decision.output_format
    elif output_format is not None:
        output_format = output_format.lower().strip()
    else:
        output_format = conversation.get("pending_output_format") or "mp3"

    if (wants_upload or wants_conversion or wants_transcription or file is not None) and output_format not in ALLOWED_OUTPUT_FORMATS:
        return chat_reply(conversation_id, f"Unsupported output format. Choose one of: {sorted(ALLOWED_OUTPUT_FORMATS)}.")

    if file is None and not url:
        if wants_upload or wants_transcription or wants_conversion:
            conversation["pending_task"] = (
                conversation.get("pending_task") or "upload"
            ) if wants_upload else intent
            conversation["pending_output_format"] = output_format
            conversation["awaiting_media"] = True
        if wants_upload or wants_transcription:
            return chat_reply(
                conversation_id,
                "Choose an audio or video file to upload.",
                action="open_upload",
            )
        if wants_conversion:
            return chat_reply(
                conversation_id,
                "Please upload a video or send a video URL. You can also choose an output format.",
                action="open_upload",
            )
        reply = decision.reply or "I can transcribe audio, convert videos, and report media-job status. How can I help?"
        return chat_reply(conversation_id, reply)

    if has_active_job():
        return chat_reply(
            conversation_id,
            "A previous media job is still processing. Please wait for it to finish before starting another one.",
            job_id=get_active_job_id(),
            status="processing",
        )

    job_id = uuid4().hex
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir()
    claim_job(job_id, conversation_id)
    conversation["last_job_id"] = job_id

    if url:
        try:
            validated_url = validate_media_url(url)
        except HTTPException:
            jobs[job_id] = {"status": "failed", "error": "Invalid media URL."}
            release_job()
            raise
        submit_job(
            run_url_conversion_job, job_id, validated_url, job_dir, output_format
        )
        clear_pending_task(conversation)
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
    input_path = job_dir / (filename or "uploaded_media")
    try:
        await save_upload(file, input_path)
    except Exception as exc:
        jobs[job_id] = {"status": "failed", "error": str(exc)}
        release_job()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    attachment = {
        "filename": filename,
        "size_bytes": input_path.stat().st_size,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
    if wants_conversion:
        submit_job(
            run_uploaded_conversion_job, job_id, input_path, output_format
        )
        reply = "Your video is processing. I will let you know when the converted audio and transcript are ready."
    else:
        submit_job(run_transcription_job, job_id, input_path)
        reply = "Your transcription is processing. I will let you know when it is ready."

    clear_pending_task(conversation)

    return chat_reply(
        conversation_id,
        reply,
        job_id=job_id,
        status="processing",
        attachment=attachment,
    )


def clear_pending_task(conversation: dict) -> None:
    for key in ("pending_task", "pending_output_format", "awaiting_media"):
        conversation.pop(key, None)
    conversation["job_status"] = "processing"

@app.post("/convert/url")
async def convert_video_url(
    background_tasks: BackgroundTasks,
    url: str = Form(...),
    output_format: str = Form("mp3"),
):
    output_format = output_format.lower().strip()
    if output_format not in ALLOWED_OUTPUT_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported output format. Choose one of: {sorted(ALLOWED_OUTPUT_FORMATS)}",
        )

    url = validate_media_url(url)
    if has_active_job():
        raise HTTPException(
            status_code=409,
            detail="Another media job is already in process. Please wait for it to finish.",
        )

    job_id = uuid4().hex
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir()
    claim_job(job_id)
    submit_job(
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
    output_format = output_format.lower().strip()

    if output_format not in ALLOWED_OUTPUT_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported output format. Choose one of: {sorted(ALLOWED_OUTPUT_FORMATS)}",
        )

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename was provided.")

    if has_active_job():
        raise HTTPException(
            status_code=409,
            detail="Another media job is already in process. Please wait for it to finish.",
        )

    job_id = uuid4().hex
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir()
    input_path = job_dir / f"input{Path(file.filename).suffix or '.bin'}"
    claim_job(job_id)

    try:
        await save_upload(file, input_path)
        submit_job(
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
        release_job()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/transcribe")
async def transcribe_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    if has_active_job():
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
    claim_job(job_id)
    try:
        await save_upload(file, audio_path)
        submit_job(run_transcription_job, job_id, audio_path)
        return {
            "job_id": job_id,
            "status": "processing",
            "message": "Transcription started.",
        }
    except Exception as exc:
        jobs[job_id] = {"status": "failed", "error": str(exc)}
        release_job()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {"job_id": job_id, **job}

@app.get("/conversations/{conversation_id}/files")
async def get_conversation_files(conversation_id: str):
    return get_files(conversation_id)
