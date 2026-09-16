from pathlib import Path
from typing import Any, cast

import yt_dlp


class MediaDownloadError(RuntimeError):
    """Raised when a video URL cannot be downloaded into a job directory."""


def download_media(url: str, output_dir: Path) -> Path:
    """Download the best available audio stream for one video URL."""
    options: dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": str(output_dir / "source.%(ext)s"),
        "noplaylist": True,
        "socket_timeout": 30,
        "retries": 2,
        "max_filesize": 1_000_000_000,
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(cast(Any, options)) as downloader:
            info = downloader.extract_info(url, download=True)
            requested_downloads = info.get("requested_downloads") or []

            if requested_downloads:
                downloaded_path = Path(requested_downloads[0]["filepath"])
            else:
                downloaded_path = Path(downloader.prepare_filename(info))
    except Exception as exc:
        raise MediaDownloadError(f"Unable to download media from the supplied URL: {exc}") from exc

    if not downloaded_path.exists():
        raise MediaDownloadError("The media download finished without creating a file.")

    return downloaded_path
