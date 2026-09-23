import asyncio
import os
import re
from pathlib import Path
from typing import Any

import gradio as gr
import httpx
from gradio.components.markdown import Markdown
from gradio.routes import mount_gradio_app

from app import app  # Existing FastAPI application instance

BASE_DIR = Path(__file__).resolve().parent
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
DEFAULT_OUTPUT_FORMAT = "mp3"

URL_REGEX = (
    r"https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\."
    r"[a-zA-Z0-9()]{1,6}\b"
    r"(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
)

PAGE_CSS = """
html, body {
    height: 100%;
    margin: 0;
    padding: 0;
    overflow: hidden !important;
}

body {
    min-height: 100vh;
}

.gradio-container {
    height: 100vh !important;
    height: 100dvh !important;
    min-height: 0 !important;
    overflow: hidden !important;
    padding-bottom: 0 !important;
}

/* Main Voxera flex column */
#voxera-app {
    height: 100% !important;
    min-height: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
    gap: 0 !important;
}

/* Header must never consume flexible height */
#voxera-header {
    flex: 0 0 auto !important;
    min-height: 0 !important;
}

/* The Chatbot is the ONLY scrolling area */
#voxera-chatbot {
    flex: 1 1 auto !important;
    min-height: 0 !important;
    height: auto !important;
    max-height: none !important;
    overflow: hidden !important;
}

/* Make Gradio's internal chatbot wrapper fill its allocated height */
#voxera-chatbot > div,
#voxera-chatbot .wrap,
#voxera-chatbot .bubble-wrap {
    min-height: 0 !important;
}

/* Composer never gets pushed below the viewport */
#voxera-composer {
    flex: 0 0 auto !important;
    min-height: 0 !important;
    overflow: visible !important;
    padding-top: 8px !important;
    padding-bottom: 8px !important;
}

/* Keep the input row compact and usable */
#voxera-input-row {
    align-items: center !important;
    gap: 8px !important;
}

#voxera-message {
    min-width: 0 !important;
}

/* Attachment button */
#voxera-upload button {
    min-width: 52px !important;
    width: 52px !important;
    height: 52px !important;
    border-radius: 14px !important;
    font-size: 20px !important;
}

/* Send button */
#voxera-send {
    min-width: 52px !important;
    width: 52px !important;
    height: 52px !important;
    border-radius: 14px !important;
}

/* Selected-file line should not push the composer off screen */
#voxera-file-status {
    min-height: 20px !important;
    max-height: 28px !important;
    overflow: hidden !important;
    margin: 0 !important;
    padding: 0 4px !important;
}

/* Hide the default footer so it cannot steal vertical space */
footer {
    display: none !important;
}
"""
def as_dict(value: Any) -> dict[str, Any]:
    """Return a dictionary only when the backend actually returned one."""
    return value if isinstance(value, dict) else {}


def normalize_uploaded_file(uploaded: Any) -> str | None:
    """Get a local filepath from Gradio UploadButton output."""
    if not uploaded:
        return None

    if isinstance(uploaded, (list, tuple)):
        if not uploaded:
            return None
        uploaded = uploaded[0]

    if isinstance(uploaded, str):
        return uploaded

    if isinstance(uploaded, dict):
        path = uploaded.get("path") or uploaded.get("name")
        return str(path) if path else None

    path = getattr(uploaded, "path", None) or getattr(uploaded, "name", None)
    return str(path) if path else None

def set_last_assistant(history: list[dict[str, Any]], content: str) -> None:
    if history:
        history[-1]["content"] = content

def job_message(job_id: str, reply: str) -> str:
    return f"⏳ **{reply}**\n\n*Job ID:* `{job_id}`\n\n<!-- job:{job_id} -->"

def find_job_message(history: list[dict[str, Any]], job_id: str) -> int | None:
    marker = f"<!-- job:{job_id} -->"
    for index in range(len(history) - 1, -1, -1):
        if marker in str(history[index].get("content", "")):
            return index
    return None

def handle_upload(uploaded: Any):
    """Store the selected file until the user sends the message."""
    file_path = normalize_uploaded_file(uploaded)

    if not file_path:
        return None, ""

    return file_path, f"📎 **{Path(file_path).name}** attached"

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

    file_handles = []

    try:
        # The current backend accepts one uploaded file for /chat.
        if file_paths:
            primary_file = file_paths[0]
            fh = open(primary_file, "rb")
            file_handles.append(fh)
            form_fields["file"] = (
                os.path.basename(primary_file),
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

        return response.status_code, as_dict(data), response.text

    finally:
        for handle in file_handles:
            handle.close()

async def chat_pipeline(
    text,
    pending_file,
    output_format,
    conversation_id,
    history,
    active_jobs,
):
    text = str(text or "").strip()
    history = list(history or [])
    active_jobs = list(active_jobs or [])
    output_format = output_format or DEFAULT_OUTPUT_FORMAT

    file_paths: list[str] = []
    if pending_file:
        file_paths.append(str(pending_file))

    if not text and not file_paths:
        yield (
            history,
            conversation_id,
            active_jobs,
            "",
            None,
            "",
        )
        return

    for file_path in file_paths:
        history.append(
            {
                "role": "user",
                "content": {"path": file_path},
            }
        )

    if text:
        history.append({"role": "user", "content": text})

    history.append({"role": "assistant", "content": "Thinking..."})

    yield (
        history,
        conversation_id,
        active_jobs,
        "",
        None,
        "",
    )

    extracted_url = None
    if text:
        match = re.search(URL_REGEX, text)
        if match:
            extracted_url = match.group(0)

    try:
        status_code, data, response_text = await post_chat(
            text=text,
            output_format=output_format,
            conversation_id=conversation_id,
            file_paths=file_paths,
            extracted_url=extracted_url,
        )
    except Exception as exc:
        set_last_assistant(history, f"❌ **Network Error:** {exc}")
        yield (
            history,
            conversation_id,
            active_jobs,
            "",
            None,
            "",
        )
        return

    if status_code != 200:
        detail = data.get("detail", response_text)
        set_last_assistant(history, f"❌ **Backend Error:** {detail}")
        yield (
            history,
            conversation_id,
            active_jobs,
            "",
            None,
            "",
        )
        return

    conversation_id = data.get("conversation_id", conversation_id)
    job_id = data.get("job_id")
    initial_reply = data.get("reply", "Processing...")
    action = data.get("action")

    if action == "open_upload":
        set_last_assistant(history, initial_reply)
        history.append(
            {
                "role": "assistant",
                "content": "📎 Use the attachment button below to choose an audio/video file.",
            }
        )
        yield (
            history,
            conversation_id,
            active_jobs,
            "",
            None,
            "",
        )
        return

    if job_id:
        active_jobs.append(job_id)
        set_last_assistant(history, job_message(job_id, initial_reply))
        yield (
            history,
            conversation_id,
            active_jobs,
            "",
            None,
            "",
        )
        return

    set_last_assistant(history, initial_reply)
    yield (
        history,
        conversation_id,
        active_jobs,
        "",
        None,
        "",
    )


# -----------------------------------------------------------------------------
# Background job polling
# -----------------------------------------------------------------------------
async def fetch_job(client: httpx.AsyncClient, job_id: str):
    try:
        response = await client.get(f"{API_BASE_URL}/jobs/{job_id}")

        try:
            data = response.json()
        except Exception:
            data = {
                "status": "failed",
                "error": response.text,
            }

        return job_id, as_dict(data)

    except Exception:
        return job_id, None


async def poll_jobs(active_jobs, history):
    active_jobs = list(active_jobs or [])
    history = list(history or [])

    if not active_jobs or not history:
        return history, active_jobs

    async with httpx.AsyncClient(timeout=30.0) as client:
        results = await asyncio.gather(
            *(fetch_job(client, job_id) for job_id in active_jobs)
        )

    remaining_jobs: list[str] = []

    for job_id, job_data in results:
        if not job_data:
            remaining_jobs.append(job_id)
            continue

        target_idx = find_job_message(history, job_id)
        status = job_data.get("status")

        if status == "processing":
            remaining_jobs.append(job_id)
            continue

        if target_idx is None:
            if status not in {"done", "failed"}:
                remaining_jobs.append(job_id)
            continue

        if status == "done":
            # This is the defensive fix for the VS Code/Pylance .get warning
            # around the old result.get(...) lines. API data is external input,
            # so never assume "result" is a dictionary.
            result = as_dict(job_data.get("result"))

            audio_url = result.get("audio_url")
            transcript_url = result.get("transcript_url")
            audio_name = result.get("audio_filename", "audio.mp3")
            transcript_name = result.get(
                "transcript_filename",
                "transcript.txt",
            )
            message = result.get("message", "Files ready")

            markdown_payload = f"### ✅ {message}\n\n"

            if transcript_url:
                markdown_payload += (
                    f"📄 **Transcript:** "
                    f"[{transcript_name}]({transcript_url})\n\n"
                )

            if audio_url:
                markdown_payload += (
                    f"🎧 **Extracted Audio:** "
                    f"[{audio_name}]({audio_url})\n\n"
                )

            history[target_idx]["content"] = markdown_payload
            continue

        if status == "failed":
            err_msg = job_data.get(
                "error",
                "Unknown processing failure occurred.",
            )
            history[target_idx]["content"] = (
                f"❌ **Job Failed:** {err_msg}"
            )
            continue

        remaining_jobs.append(job_id)

    return history, remaining_jobs

def build_voxera_ui():
    with gr.Blocks(
        title="Voxera AI",
        fill_height=True,
        css=PAGE_CSS,
    ) as demo:
        conversation_state = gr.State(value=None)
        active_jobs_state = gr.State(value=[])
        output_format_state = gr.State(value=DEFAULT_OUTPUT_FORMAT)
        pending_file_state = gr.State(value=None)

        with gr.Column(
            elem_id="voxera-app",
            variant="panel",
            scale=1,
        ):
            with gr.Row(
                elem_id="voxera-header",
                scale=0,
            ):
                Markdown(
                    "## 🎙️ Voxera\n"
                    "Conversational media extraction & transcription"
                )
                clear_btn = gr.Button(
                    "Reset",
                    size="sm",
                    variant="secondary",
                    scale=0,
                    min_width=90,
                )

            chatbot = gr.Chatbot(
                elem_id="voxera-chatbot",
                scale=1,
                min_height=0,
                height=None,
                show_label=False,
                autoscroll=True,
                avatar_images=(
                    None,
                    "https://api.dicebear.com/7.x/bottts/svg?seed=Voxera",
                ),
                placeholder=(
                    "### ✦ Talk to Voxera\n"
                    "Ask a question, upload media, or paste a video URL."
                ),
            )

            with gr.Column(
                elem_id="voxera-composer",
                scale=0,
            ):
                file_status = Markdown(
                    "",
                    elem_id="voxera-file-status",
                    show_label=False,
                )

                with gr.Row(
                    elem_id="voxera-input-row",
                    scale=0,
                ):
                    upload_btn = gr.UploadButton(
                        "📎",
                        elem_id="voxera-upload",
                        file_types=["audio", "video"],
                        file_count="single",
                        type="filepath",
                        size="lg",
                        variant="secondary",
                        scale=0,
                    )

                    chat_input = gr.Textbox(
                        elem_id="voxera-message",
                        placeholder=(
                            "Message Voxera, paste a link, "
                            "or attach audio/video…"
                        ),
                        show_label=False,
                        lines=1,
                        max_lines=6,
                        submit_btn=False,
                        scale=1,
                    )

                    send_btn = gr.Button(
                        "➤",
                        elem_id="voxera-send",
                        variant="primary",
                        size="lg",
                        scale=0,
                    )

                Markdown(
                    "Enter to send · 📎 choose a file · paste a link",
                    elem_id="voxera-hint",
                )

        poll_timer = gr.Timer(2.0)

        upload_btn.upload(
            fn=handle_upload,
            inputs=upload_btn,
            outputs=[pending_file_state, file_status],
        )

        submit_inputs = [
            chat_input,
            pending_file_state,
            output_format_state,
            conversation_state,
            chatbot,
            active_jobs_state,
        ]

        submit_outputs = [
            chatbot,
            conversation_state,
            active_jobs_state,
            chat_input,
            pending_file_state,
            file_status,
        ]

        # Enter in the textbox
        chat_input.submit(
            fn=chat_pipeline,
            inputs=submit_inputs,
            outputs=submit_outputs,
        )

        # Click the send button
        send_btn.click(
            fn=chat_pipeline,
            inputs=submit_inputs,
            outputs=submit_outputs,
        )

        # Poll background transcription/extraction jobs.
        poll_timer.tick(
            fn=poll_jobs,
            inputs=[active_jobs_state, chatbot],
            outputs=[chatbot, active_jobs_state],
            queue=False,
        )

        def reset_session():
            return [], None, [], "", None, ""

        clear_btn.click(
            fn=reset_session,
            inputs=None,
            outputs=[
                chatbot,
                conversation_state,
                active_jobs_state,
                chat_input,
                pending_file_state,
                file_status,
            ],
            queue=False,
        )

    return demo

voxera_ui = build_voxera_ui()

app = mount_gradio_app(
    app,
    voxera_ui,
    path="/",
)
