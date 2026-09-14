from pathlib import Path
import subprocess

def _run_ffmpeg(command: list[str]) -> None:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "FFmpeg was not found. Install FFmpeg and make sure it is available in PATH."
        ) from exc

    if result.returncode != 0:
        error = result.stderr.strip() or result.stdout.strip() or "Unknown FFmpeg error."
        raise RuntimeError(f"FFmpeg failed:\n{error}")

def convert_video_to_audio(
    input_path: Path,
    output_dir: Path,
    output_format: str,
) -> Path:
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"{input_path.stem}_{output_format}.{output_format}"

    # Basic codec/container choices for this first version.
    codec_args = {
        "mp3": ["-vn", "-codec:a", "libmp3lame", "-q:a", "2"],
        "wav": ["-vn", "-codec:a", "pcm_s16le"],
        "m4a": ["-vn", "-codec:a", "aac", "-b:a", "192k"],
        "flac": ["-vn", "-codec:a", "flac"],
        "aac": ["-vn", "-codec:a", "aac", "-b:a", "192k", "-f", "adts"],
        "ogg": ["-vn", "-codec:a", "libvorbis", "-q:a", "5"],
        "opus": ["-vn", "-codec:a", "libopus", "-b:a", "160k"],
    }

    if output_format not in codec_args:
        raise ValueError(f"Unsupported output format: {output_format}")

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        *codec_args[output_format],
        str(output_path),
    ]
    _run_ffmpeg(command)

    if not output_path.exists():
        raise RuntimeError("FFmpeg completed but the output file was not created.")

    return output_path