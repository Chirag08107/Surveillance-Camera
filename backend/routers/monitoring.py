
import mimetypes
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

router = APIRouter(prefix="/api/monitoring", tags=["Monitoring"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = PROJECT_ROOT / "uploads"
LOG_DIR = UPLOAD_DIR / "logs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024

_process = None
_current_video = None
_current_video_name = None
_current_session_id = None
_log_handle = None
_lock = threading.Lock()


def _process_running():
    return _process is not None and _process.poll() is None


def _close_log():
    global _log_handle

    if _log_handle is not None:
        try:
            _log_handle.close()
        except OSError:
            pass
        _log_handle = None


def _read_log_tail(path, limit=4000):
    try:
        return path.read_text(
            encoding="utf-8",
            errors="replace",
        )[-limit:]
    except OSError:
        return ""


def _session_status():
    running = _process_running()

    return {
        "running": running,
        "session_id": _current_session_id,
        "video_name": _current_video_name,
        "video_url": (
            f"/api/monitoring/video/{_current_video.name}"
            if _current_video is not None and _current_video.exists()
            else None
        ),
        "return_code": (
            _process.returncode
            if _process is not None and not running
            else None
        ),
        "message": (
            "Monitoring pipeline is running"
            if running
            else "No monitoring pipeline is running"
        ),
    }


@router.get("/status")
def monitoring_status():
    with _lock:
        if _process is not None and _process.poll() is not None:
            _close_log()

        return _session_status()


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    global _process
    global _current_video
    global _current_video_name
    global _current_session_id
    global _log_handle

    original_name = Path(file.filename or "").name
    extension = Path(original_name).suffix.lower()

    if not original_name or extension not in ALLOWED_EXTENSIONS:
        await file.close()
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported video format. "
                "Upload MP4, AVI, MOV, MKV, or WEBM."
            ),
        )

    # Check the running process before accepting a new session.
    with _lock:
        if _process_running():
            await file.close()
            raise HTTPException(
                status_code=409,
                detail=(
                    "A monitoring session is already running. "
                    "Stop it before uploading another video."
                ),
            )

        _close_log()

        session_id = str(uuid.uuid4())
        saved_path = UPLOAD_DIR / f"{session_id}{extension}"
        log_path = LOG_DIR / f"{session_id}.log"
        total_bytes = 0

        try:
            with saved_path.open("wb") as output:
                while True:
                    chunk = await file.read(1024 * 1024)

                    if not chunk:
                        break

                    total_bytes += len(chunk)

                    if total_bytes > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail="Video exceeds the 500 MB upload limit.",
                        )

                    output.write(chunk)

        except HTTPException:
            saved_path.unlink(missing_ok=True)
            raise

        except Exception as exc:
            saved_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=500,
                detail=f"Could not save uploaded video: {exc}",
            ) from exc

        finally:
            await file.close()

        if total_bytes == 0:
            saved_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=400,
                detail="The uploaded video is empty.",
            )

        pipeline_path = PROJECT_ROOT / "pipeline.py"

        if not pipeline_path.exists():
            saved_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=500,
                detail=f"Pipeline file was not found: {pipeline_path}",
            )

        try:
            _log_handle = log_path.open("w", encoding="utf-8")

            # Pass the session ID to the child process.
            # pipeline.py must read this variable and include the ID
            # in its backend ingest requests for session filtering.
            pipeline_env = os.environ.copy()
            pipeline_env["SURVEILLANCE_SESSION_ID"] = session_id

            _process = subprocess.Popen(
                [
                    sys.executable,
                    str(pipeline_path),
                    "--video",
                    str(saved_path),
                    "--backend-url",
                    "http://127.0.0.1:8000",
                    "--headless",
                ],
                cwd=str(PROJECT_ROOT),
                stdin=subprocess.DEVNULL,
                stdout=_log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=pipeline_env,
            )

            # Detect immediate startup failures.
            time.sleep(1)

            if _process.poll() is not None:
                return_code = _process.returncode
                _close_log()
                log_text = _read_log_tail(log_path)

                _process = None
                saved_path.unlink(missing_ok=True)

                raise HTTPException(
                    status_code=500,
                    detail=(
                        "The monitoring pipeline exited during startup. "
                        f"Exit code: {return_code}. "
                        f"Recent pipeline log: {log_text[-2000:]}"
                    ),
                )

        except HTTPException:
            raise

        except Exception as exc:
            _close_log()
            _process = None
            saved_path.unlink(missing_ok=True)

            raise HTTPException(
                status_code=500,
                detail=f"Could not start the pipeline: {exc}",
            ) from exc

        _current_video = saved_path
        _current_video_name = original_name
        _current_session_id = session_id

        return {
            "status": "started",
            "session_id": session_id,
            "video_name": original_name,
            "video_url": f"/api/monitoring/video/{saved_path.name}",
            "message": "Video uploaded and monitoring pipeline started.",
        }


@router.get("/video/{filename}")
def serve_uploaded_video(filename: str):
    with _lock:
        if (
            _current_video is None
            or not _current_video.exists()
            or filename != _current_video.name
        ):
            raise HTTPException(
                status_code=404,
                detail="Video not found for the current monitoring session.",
            )

        video_path = _current_video
        video_name = _current_video_name or video_path.name

    media_type = mimetypes.guess_type(video_path.name)[0]
    if not media_type:
        media_type = "application/octet-stream"

    return FileResponse(
        path=video_path,
        filename=video_name,
        media_type=media_type,
        content_disposition_type="inline",
    )


@router.post("/stop")
def stop_monitoring():
    global _process
    global _log_handle

    with _lock:
        if not _process_running():
            _close_log()
            return {
                "status": "stopped",
                "message": "No monitoring pipeline is running.",
            }

        process = _process

        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)

        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

            process.wait(timeout=3)

        except ProcessLookupError:
            pass

        finally:
            _process = None
            _close_log()

        return {
            "status": "stopped",
            "message": "Monitoring pipeline stopped.",
        }
