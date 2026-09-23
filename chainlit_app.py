import asyncio
import os
import re
from typing import Any

import chainlit as cl
import httpx

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
DEFAULT_OUTPUT_FORMAT = "mp3"

URL_REGEX = (
    r"https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\."
    r"[a-zA-Z0-9()]{1,6}\b"
    r"(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
)


def extract_file_paths(message: cl.Message) -> list[str]:
    paths: list[str] = []
    for element in message.elements or []:
        path = getattr(element, "path", None)
        mime = getattr(element, "mime", "") or ""
        if path and (mime.startswith("audio/") or mime.startswith("video/")):
            paths.append(str(path))
    return paths


async def post_chat(
    text: str,
    output_format: str,
    conversation_id: str | None,
    file_paths: list[str],
    extracted_url: str | None,
):
    form_fields: dict[str, Any] = {
        "message": (None, text),
        "output_format": (None, output_format),
    }

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
                "application/octet-stream",
            )

        async with httpx.AsyncClient(timeout=60.0) as client:
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
        return "**Job finished, but the backend returned an unexpected result.**"

    audio_url = result.get("audio_url")
    transcript_url = result.get("transcript_url")
    audio_name = result.get("audio_filename", "audio.mp3")
    transcript_name = result.get("transcript_filename", "transcript.txt")
    message = result.get("message", "Files ready")

    output = f"### ✅ {message}\n\n"
    if transcript_url:
        output += f"📄 **Transcript:** [{transcript_name}]({transcript_url})\n\n"
    if audio_url:
        output += f"🎧 **Extracted Audio:** [{audio_name}]({audio_url})\n\n"
    return output


async def update_message(message: cl.Message, content: str) -> None:
    """Set message content before asking Chainlit to refresh it."""
    message.content = content
    await message.update()


async def poll_job(job_id: str, assistant_message: cl.Message):
    while True:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{API_BASE_URL}/jobs/{job_id}")

            try:
                data = response.json()
            except Exception:
                await update_message(assistant_message, f"❌ **Job Error:** {response.text}")
                return

            status = data.get("status")

            if status == "processing":
                await asyncio.sleep(2)
                continue

            if status == "done":
                await update_message(assistant_message, format_job_result(data))
                return

            if status == "failed":
                error = data.get("error", "Unknown processing failure occurred.")
                await update_message(assistant_message, f"❌ **Job Failed:** {error}")
                return

            await update_message(assistant_message, f"❌ **Unknown job status:** `{status}`")
            return

        except Exception as exc:
            await update_message(assistant_message, f"❌ **Polling Error:** {exc}")
            return


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("conversation_id", None)


@cl.on_message
async def on_message(message: cl.Message):
    text = (message.content or "").strip()
    file_paths = extract_file_paths(message)

    if not text and not file_paths:
        await cl.Message(
            content="Please type a message, upload an audio/video file, or paste a URL."
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
        await update_message(assistant_message, f"❌ **Network Error:** {exc}")
        return

    if status_code != 200:
        detail = (
            data.get("detail", response_text)
            if isinstance(data, dict)
            else response_text
        )
        await update_message(assistant_message, f"❌ **Backend Error:** {detail}")
        return

    if isinstance(data, dict):
        cl.user_session.set(
            "conversation_id",
            data.get("conversation_id", conversation_id),
        )

    initial_reply = data.get("reply", "Processing...") if isinstance(data, dict) else "Processing..."
    action = data.get("action") if isinstance(data, dict) else None
    job_id = data.get("job_id") if isinstance(data, dict) else None

    if action == "open_upload":
        await update_message(
            assistant_message,
            (
                f"{initial_reply}\n\n"
                "📎 **Use the attachment button in the message bar "
                "to choose an audio/video file from your computer.**"
            )
        )
        return

    if job_id:
        await update_message(
            assistant_message,
            f"⏳ **{initial_reply}**\n\n*Job ID:* `{job_id}`",
        )

        # Do not await: the user can continue chatting while this job runs.
        asyncio.create_task(poll_job(job_id, assistant_message))
        return

    await update_message(assistant_message, initial_reply)
