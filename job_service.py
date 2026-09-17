from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from file_service import upload_and_register
from media_download import download_media
from transcribe import transcribe_audio
from video_audio import convert_video_to_audio

jobs = {}
_active_job_id = None
_job_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="media-job")


def submit_job(job_function, *args):
    _job_executor.submit(job_function, *args)

def has_active_job() -> bool:
    return _active_job_id is not None

def get_active_job_id():
    return _active_job_id

def claim_job(job_id: str, conversation_id: str | None = None):
    global _active_job_id
    _active_job_id = job_id
    job = {"status": "processing"}
    if conversation_id:
        job["conversation_id"] = conversation_id
    jobs[job_id] = job

def release_job():
    global _active_job_id
    _active_job_id = None

def write_transcript(path: Path, text: str):
    path.write_text(text, encoding="utf-8")

def job_result(audio_path: Path, transcript_path: Path, audio_url: str, transcript_url: str):
    return {
        "message": "Files ready",
        "audio_url": audio_url,
        "transcript_url": transcript_url,
        "audio_filename": audio_path.name,
        "transcript_filename": transcript_path.name,
    }

def transcribe_and_upload(
    job_id: str,
    audio_path: Path,
    conversation_id: str | None = None,
    source_path: Path | None = None,
):
    transcript_path = audio_path.with_suffix(".txt")
    transcript = transcribe_audio(str(audio_path))
    write_transcript(transcript_path, transcript)
    audio_url = upload_and_register(conversation_id, job_id, audio_path, "audio")
    transcript_url = upload_and_register(
        conversation_id, job_id, transcript_path, "transcript"
    )
    if conversation_id and source_path and source_path.resolve() != audio_path.resolve():
        upload_and_register(conversation_id, job_id, source_path, "video")
    return job_result(audio_path, transcript_path, audio_url, transcript_url)

def finish_job(job_id: str, audio_path: Path, source_path: Path | None = None):
    job = jobs[job_id]
    result = transcribe_and_upload(
        job_id,
        audio_path,
        job.get("conversation_id"),
        source_path,
    )
    jobs[job_id] = {"status": "done", "result": result}

def _run_job(job_id: str, work):
    try:
        work()
    except Exception as exc:
        jobs[job_id] = {"status": "failed", "error": str(exc)}
    finally:
        release_job()

def run_transcription_job(job_id: str, audio_path: Path):
    _run_job(job_id, lambda: finish_job(job_id, audio_path))

def run_uploaded_conversion_job(job_id: str, input_path: Path, output_format: str):
    def work():
        audio_path = convert_video_to_audio(
            input_path=input_path,
            output_dir=input_path.parent,
            output_format=output_format,
        )
        finish_job(job_id, audio_path, input_path)
        input_path.unlink(missing_ok=True)

    _run_job(job_id, work)

def run_url_conversion_job(
    job_id: str,
    url: str,
    job_dir: Path,
    output_format: str,
):
    def work():
        input_path = download_media(url, job_dir)
        audio_path = convert_video_to_audio(
            input_path=input_path,
            output_dir=job_dir,
            output_format=output_format,
        )
        finish_job(job_id, audio_path, input_path)
        input_path.unlink(missing_ok=True)

    _run_job(job_id, work)