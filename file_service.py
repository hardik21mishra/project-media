from datetime import datetime, timezone
from pathlib import Path
from drive_upload import upload_to_drive

conversations = {}

def register_file(
    conversation_id: str | None,
    job_id: str,
    path: Path,
    file_type: str,
    url: str,
):
    if not conversation_id:
        return

    conversation = conversations.setdefault(
        conversation_id,
        {"messages": [], "files": []},
    )
    conversation.setdefault("files", []).append(
        {
            "job_id": job_id,
            "filename": path.name,
            "file_type": file_type,
            "url": url,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )

def upload_and_register(
    conversation_id: str | None,
    job_id: str,
    path: Path,
    file_type: str,
):
    url = upload_to_drive(str(path))
    register_file(conversation_id, job_id, path, file_type, url)
    return url

def get_files(conversation_id: str):
    conversation = conversations.get(conversation_id)
    if conversation is None:
        return {"conversation_id": conversation_id, "files": []}
    return {
        "conversation_id": conversation_id,
        "files": conversation.get("files", []),
    }