import asyncio
from contextvars import copy_context
import mimetypes
import logging
import os
import re
import tomllib
from pathlib import Path
from typing import Any

import chainlit as cl
from chainlit.context import local_steps
import httpx
from chainlit.config import FILES_DIRECTORY, SpontaneousFileUploadFeature, config as chainlit_config

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
DEFAULT_OUTPUT_FORMAT = "mp3"
COMPLETION_NOTIFICATIONS: set[str] = set()
logger = logging.getLogger(__name__)

UPLOAD_CONFIG_PATH = Path(__file__).resolve().with_name("config_uploads.toml")
with UPLOAD_CONFIG_PATH.open("rb") as config_file:
    upload_config = tomllib.load(config_file)

upload_settings = SpontaneousFileUploadFeature.model_validate(
    upload_config["features"]["spontaneous_file_upload"]
)
if (
    not upload_settings.accept
    or not upload_settings.max_files
    or not upload_settings.max_size_mb
):
    raise ValueError(f"Incomplete upload settings in {UPLOAD_CONFIG_PATH}")
if upload_settings.max_files != 1 or upload_settings.max_size_mb < 1:
    raise ValueError(f"The /chat endpoint accepts one file; check {UPLOAD_CONFIG_PATH}")

UPLOAD_TIMEOUT_SECONDS = upload_config["bot_upload"]["timeout_seconds"]
REQUEST_TIMEOUT_SECONDS = upload_config["backend"]["request_timeout_seconds"]
if any(
    type(value) is not int or value < 1
    for value in (UPLOAD_TIMEOUT_SECONDS, REQUEST_TIMEOUT_SECONDS)
):
    raise ValueError(f"Upload timeouts must be positive integers in {UPLOAD_CONFIG_PATH}")

# AskFileMessage uses Chainlit's spontaneous upload feature and rejects uploads
# when it is disabled. Apply the configured limits with uploads enabled.
chainlit_config.features.spontaneous_file_upload = upload_settings

# Used by the normal message composer/sidebar upload.
MEDIA_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".opus",
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".mpeg", ".mpg",
    ".m4v", ".3gp",
}

UPLOAD_ACCEPT = upload_settings.accept
UPLOAD_MAX_FILES = upload_settings.max_files
UPLOAD_MAX_SIZE_MB = upload_settings.max_size_mb

URL_REGEX = (
    r"https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\."
    r"[a-zA-Z0-9()]{1,6}\b"
    r"(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
)

# check if the uploaded file is audio or a video
def is_media_file(path: str, mime: str = "") -> bool:
    mime = (mime or "").lower()
    extension = os.path.splitext(path)[1].lower()

    return (
        mime.startswith("audio/")
        or mime.startswith("video/")
        or extension in MEDIA_EXTENSIONS
    )

def extract_file_paths(message: cl.Message) -> list[str]:
    paths: list[str] = []

    for element in message.elements or []:
        path = getattr(element, "path", None)
        mime = getattr(element, "mime", "") or ""

        if path and is_media_file(str(path), str(mime)):
            paths.append(str(path))
    return paths

#song.mp3 -> audio/mpeg
#video.mp4 -> video/mp4
def get_file_mime_type(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"

async def post_chat(
    text: str,
    output_format: str,
    conversation_id: str | None,
    file_paths: list[str],
    extracted_url: str | None,
):
    form_fields: dict[str, Any] = {
        "message": (None, text),
    }
    # Let the backend remember a format selected in chat. Its default is mp3.
    if output_format != DEFAULT_OUTPUT_FORMAT:
        form_fields["output_format"] = (None, output_format)

    if conversation_id:
        form_fields["conversation_id"] = (None, conversation_id)

    if extracted_url:
        form_fields["url"] = (None, extracted_url)

    handles = []

    try:
        if file_paths:
            path = file_paths[0]
            fh = open(path, "rb")
            handles.append(fh)

            form_fields["file"] = (
                os.path.basename(path),
                fh,
                get_file_mime_type(path),
            )

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(REQUEST_TIMEOUT_SECONDS, connect=10.0)
        ) as client:
            response = await client.post(
                f"{API_BASE_URL}/chat",
                files=form_fields,
            )

        try:
            data = response.json()
        except Exception:
            data = {"detail": response.text}

        return response.status_code, data, response.text

    finally:
        for handle in handles:
            handle.close()

def format_job_result(job_data: dict[str, Any]) -> str:
    result = job_data.get("result")

    # Avoid the .get() crash when result is None/string/list.
    if not isinstance(result, dict):
        return "Job finished, but the backend returned an unexpected result."

    audio_url = result.get("audio_url")
    transcript_url = result.get("transcript_url")
    audio_name = result.get("audio_filename", "audio.mp3")
    transcript_name = result.get("transcript_filename", "transcript.txt")
    message = result.get("message", "Files ready")

    output = f"### ✅ {message}\n\n"

    if transcript_url:
        output += f"Transcript: [{transcript_name}]({transcript_url})\n\n"
    if audio_url:
        output += f"Extracted Audio: [{audio_name}]({audio_url})\n\n"
    return output

async def update_message(message: cl.Message, content: str) -> None:
    """Set message content before asking Chainlit to refresh it."""
    message.content = content
    await message.update()

async def poll_job(job_id: str, progress_message: cl.Message):
    """Poll independently and append the completed result to the chat."""
    while True:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{API_BASE_URL}/jobs/{job_id}")

            try:
                data = response.json()
            except Exception:
                await update_message(progress_message, f"Job Error: {response.text}")
                return

            if not isinstance(data, dict):
                await update_message(
                    progress_message,
                    "Job Error: the backend returned an unexpected response.",
                )
                return

            status = data.get("status")

            if status == "processing":
                await asyncio.sleep(2)
                continue
            if status == "done":
                if job_id in COMPLETION_NOTIFICATIONS:
                    return
                COMPLETION_NOTIFICATIONS.add(job_id)
                await cl.Message(content=format_job_result(data)).send()
                return
            if status == "failed":
                error = data.get(
                    "error",
                    "Unknown processing failure occurred.",
                )
                await update_message(progress_message, f"Job Failed: {error}")
                return

            await update_message(
                progress_message, f"Unknown job status: `{status}`"
            )
            return

        except Exception as exc:
            await update_message(progress_message, f"Polling Error: {exc}")
            return

async def start_job_poll(job_id: str, progress_message: cl.Message) -> None:
    """Run polling outside the message step so it cannot hold the composer busy."""
    # Chainlit's stop button is cleared by task_end. End the chat turn before
    # launching the long-lived poller so the composer is available immediately.
    session = cl.context.session
    starting_task = session.current_task
    await cl.context.emitter.task_end()
    task_context = copy_context()
    # Background updates should not become children of the completed on_message step.
    task_context.run(local_steps.set, None)

    async def clear_late_upload_task_start() -> None:
        # AskFileMessage can emit task_start while unwinding its upload prompt.
        # Clear that late signal, but never clear a newer message's loading state.
        await asyncio.sleep(0.1)
        if session.current_task is starting_task:
            await cl.context.emitter.task_end()

    asyncio.create_task(
        clear_late_upload_task_start(),
        context=task_context.copy(),
    )
    asyncio.create_task(
        poll_job(job_id, progress_message),
        context=task_context,
    )

async def submit_uploaded_file(file_path: str) -> None:
    conversation_id = cl.user_session.get("conversation_id")

    assistant_message = cl.Message(
        content=f"Upload received: `{os.path.basename(file_path)}`\n\n"
                "I’m sending it to the processor now..."
    )
    await assistant_message.send()

    try:
        status_code, data, response_text = await post_chat(
            text="",
            output_format=DEFAULT_OUTPUT_FORMAT,
            conversation_id=conversation_id,
            file_paths=[file_path],
            extracted_url=None,
        )

    except Exception as exc:
        await update_message(
            assistant_message,
            f"Network Error: {exc}",
        )
        return

    if status_code != 200:
        detail = (
            data.get("detail", response_text)
            if isinstance(data, dict)
            else response_text
        )

        await update_message(
            assistant_message,
            f"Backend Error: {detail}",
        )
        return

    if isinstance(data, dict):
        cl.user_session.set(
            "conversation_id",
            data.get("conversation_id", conversation_id),
        )

    initial_reply = (
        data.get("reply", "Processing...")
        if isinstance(data, dict)
        else "Processing..."
    )

    job_id = data.get("job_id") if isinstance(data, dict) else None

    if job_id:
        await update_message(
            assistant_message,
            f"⏳ **{initial_reply}**\n\n*Job ID:* `{job_id}`",
        )

        # Do not await: the user can continue chatting while this job runs.
        await start_job_poll(job_id, assistant_message)
        return

    await update_message(assistant_message, initial_reply)

async def open_bot_upload_ui() -> None:
    """Show the in-chat file picker and process the selected media."""
    FILES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    try:
        files = await cl.AskFileMessage(
            content=(
                "Select an audio or video file "
                "(MP3, WAV, M4A, MP4, MOV, MKV, WEBM, and more)."
            ),
            accept=UPLOAD_ACCEPT,
            max_files=UPLOAD_MAX_FILES,
            max_size_mb=UPLOAD_MAX_SIZE_MB,
            timeout=UPLOAD_TIMEOUT_SECONDS,
            raise_on_timeout=False,
        ).send()
    except Exception as exc:
        await cl.Message(content=f"Upload UI Error: {exc}").send()
        return

    if not files:
        await cl.Message(
            content="No file was selected. You can keep chatting or choose a file later."
        ).send()
        return

    selected_file = files[0]
    if not selected_file.path:
        await cl.Message(content="Upload Error: Chainlit returned no file path.").send()
        return

    await submit_uploaded_file(str(selected_file.path))

@cl.action_callback("request_media_upload")
async def request_media_upload(action: cl.Action):
    await open_bot_upload_ui()

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("conversation_id", None)

@cl.on_message
async def on_message(message: cl.Message):
    text = (message.content or "").strip()
    file_paths = extract_file_paths(message)

    if not text and not file_paths:
        await cl.Message(
            content=(
                "Please type a message, upload an audio/video file, "
                "or paste a URL."
            )
        ).send()
        return

    conversation_id = cl.user_session.get("conversation_id")
    extracted_url = None
    if text:
        match = re.search(URL_REGEX, text)

        if match:
            extracted_url = match.group(0)

    assistant_message = cl.Message(content="Thinking...")
    await assistant_message.send()

    try:
        status_code, data, response_text = await post_chat(
            text=text,
            output_format=DEFAULT_OUTPUT_FORMAT,
            conversation_id=conversation_id,
            file_paths=file_paths,
            extracted_url=extracted_url,
        )

    except Exception as exc:
        await update_message(
            assistant_message,
            f"Network Error: {exc}",
        )
        return

    if status_code != 200:
        detail = (
            data.get("detail", response_text)
            if isinstance(data, dict)
            else response_text
        )

        await update_message(
            assistant_message,
            f"Backend Error: {detail}",
        )
        return

    if isinstance(data, dict):
        cl.user_session.set(
            "conversation_id",
            data.get("conversation_id", conversation_id),
        )

    initial_reply = (
        data.get("reply", "Processing...")
        if isinstance(data, dict)
        else "Processing..."
    )

    action = data.get("action") if isinstance(data, dict) else None
    job_id = data.get("job_id") if isinstance(data, dict) else None

    if action == "open_upload":
        await update_message(assistant_message, initial_reply)
        await cl.Message(
            content=(
                "📁 Upload your media\n"
                "Choose one audio or video file when you’re ready. "
                "You can keep chatting without uploading one."
            ),
            actions=[
                cl.Action(
                    name="request_media_upload",
                    payload={},
                    label="Browse files",
                    icon="upload",
                )
            ],
        ).send()
        logger.info("chat_upload_control conversation=%s displayed=true", cl.user_session.get("conversation_id"))
        return
    if isinstance(data, dict) and data.get("status") == "done" and isinstance(data.get("result"), dict):
        await update_message(assistant_message, f"{initial_reply}\n\n{format_job_result(data)}")
        return
    if job_id and (file_paths or extracted_url):
        await update_message(
            assistant_message,
            f"{initial_reply}\n\nJob ID: `{job_id}`",
        )
        # the user can continue chatting while this job runs.
        await start_job_poll(job_id, assistant_message)
        return

    await update_message(assistant_message, initial_reply)
