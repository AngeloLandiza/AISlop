#!/usr/bin/env python3
"""
Streamlit UI for YouTube Shorts Automation.
"""

import json
import subprocess
import os
import platform
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path
from typing import Dict, Any

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yaml


PROJECT_DIR = Path(__file__).parent
CONFIG_PATH = PROJECT_DIR / "config.yaml"
CREDENTIALS_DIR = PROJECT_DIR / "credentials"
CLIENT_SECRET_PATH = CREDENTIALS_DIR / "client_secret.json"
SCHEDULE_STATE_FILE = PROJECT_DIR / "logs" / "schedule_state.json"

# Ensure UTF-8 on Windows only (macOS/Linux default to UTF-8)
if platform.system() == "Windows":
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def load_config() -> Dict[str, Any]:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def save_config(config: Dict[str, Any]) -> None:
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def check_ffmpeg() -> bool:
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def get_next_run_from_times(times: list[str]) -> datetime | None:
    if not times:
        return None
    now = datetime.now()
    candidates = []
    for t in times:
        try:
            h, m = map(int, t.split(":"))
            candidate = now.replace(hour=h, minute=m, second=0, microsecond=0)
            if candidate < now:
                candidate = candidate + timedelta(days=1)
            candidates.append(candidate)
        except Exception:
            continue
    return min(candidates) if candidates else None


def read_schedule_state() -> dict | None:
    if SCHEDULE_STATE_FILE.exists():
        try:
            with open(SCHEDULE_STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def run_command_stream(label: str, cmd: list[str], debug: bool, log_key: str) -> None:
    """Run a command and stream logs + status to the UI."""
    st.session_state["is_busy"] = True
    st.session_state[log_key] = []
    status_placeholder = st.empty()
    output_placeholder = st.empty()

    with st.status(f"{label}...", state="running") as status:
        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
            last_line = ""
            for line in iter(process.stdout.readline, ""):
                line = line.rstrip()
                if not line:
                    continue
                st.session_state[log_key].append(line)
                last_line = line
                if debug:
                    output_placeholder.code("\n".join(st.session_state[log_key][-500:]))
                else:
                    status_placeholder.info(last_line)
            process.wait()
            if process.returncode == 0:
                status.update(label=f"{label} complete", state="complete")
            else:
                status.update(label=f"{label} failed (exit {process.returncode})", state="error", expanded=True)
                output_placeholder.code("\n".join(st.session_state[log_key][-200:]))
        except Exception as e:
            status.update(label=f"{label} failed", state="error", expanded=True)
            output_placeholder.code(str(e))
        finally:
            st.session_state["is_busy"] = False


def is_authenticated() -> bool:
    from youtube_uploader import YouTubeUploader
    return YouTubeUploader().is_authenticated()


def ensure_credentials_dir() -> None:
    CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)


st.set_page_config(page_title="YouTube Shorts Automation", page_icon="🎬", layout="wide")
VERSION_FILE = PROJECT_DIR / "VERSION"
def get_version() -> str:
    try:
        return VERSION_FILE.read_text().strip()
    except Exception:
        return "0.0.0"

st.title("🎬 YouTube Shorts Automation")
st.caption(f"Version {get_version()} • Generate, review, and upload Shorts with a clean, simple UI.")

st.session_state.setdefault("debug_mode", False)
st.session_state.setdefault("logs_run", [])
st.session_state.setdefault("logs_bg", [])
st.session_state.setdefault("logs_auth", [])
st.session_state.setdefault("logs_test", [])
st.session_state.setdefault("is_busy", False)

debug_mode = st.toggle("Debug mode (show full console output)", value=st.session_state["debug_mode"])
st.session_state["debug_mode"] = debug_mode

ensure_credentials_dir()
config = load_config()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("FFmpeg", "Ready" if check_ffmpeg() else "Missing")
with col2:
    st.metric("YouTube Auth", "Ready" if is_authenticated() else "Not Authenticated")
with col3:
    provider = config.get("tts", {}).get("provider", "gtts")
    st.metric("TTS Provider", provider)

st.divider()

st.header("🔑 Credentials & API Keys")
st.markdown("All credentials live in the `credentials/` folder.")

with st.expander("Upload YouTube OAuth Client Secret", expanded=True):
    st.write("Drop your OAuth client JSON here. It will be saved as `credentials/client_secret.json`.")
    upload = st.file_uploader("client_secret.json", type=["json"])
    if upload is not None:
        data = upload.read()
        try:
            json.loads(data.decode("utf-8"))
            CLIENT_SECRET_PATH.write_bytes(data)
            st.success("Saved client_secret.json to credentials/")
        except Exception:
            st.error("Invalid JSON file. Please upload the OAuth client JSON file.")

with st.expander("API Keys", expanded=True):
    apis = config.setdefault("APIs", {})
    cohere = apis.setdefault("cohere", {})
    claude = apis.setdefault("claude", {})

    cohere_key = st.text_input("Cohere API Key", value=cohere.get("api_key", ""), type="password")
    claude_key = st.text_input("Claude API Key (optional)", value=claude.get("api_key", ""), type="password")

    if st.button("Save API Keys"):
        cohere["api_key"] = cohere_key.strip()
        claude["api_key"] = claude_key.strip()
        save_config(config)
        st.success("API keys saved to config.yaml")

st.divider()

st.header("⚙️ Settings")

with st.expander("Core Settings", expanded=True):
    tts = config.setdefault("tts", {})
    video = config.setdefault("video", {})
    schedule_cfg = config.setdefault("schedule", {})
    youtube_cfg = config.setdefault("youtube", {})

    tts_provider = st.selectbox("TTS Provider", ["gtts", "piper"], index=0 if tts.get("provider", "gtts") == "gtts" else 1)
    tts_speed = st.slider("TTS Speed", min_value=0.7, max_value=1.6, value=float(tts.get("speed", 1.0)), step=0.05)
    video_speed = st.slider("Final Video Speed", min_value=0.8, max_value=1.6, value=float(video.get("speed", 1.3)), step=0.05)
    max_duration = st.number_input("Max Duration (seconds)", min_value=15, max_value=120, value=int(video.get("max_duration", 60)))
    shorts_mode = st.checkbox("Shorts Mode (add #shorts metadata)", value=bool(youtube_cfg.get("shorts_mode", True)))

    if st.button("Save Core Settings"):
        tts["provider"] = tts_provider
        tts["speed"] = float(tts_speed)
        video["speed"] = float(video_speed)
        video["max_duration"] = int(max_duration)
        youtube_cfg["shorts_mode"] = shorts_mode
        save_config(config)
        st.success("Settings saved")

with st.expander("Schedule (Fixed or Random)"):
    st.write("Use either fixed times or random windows.")
    enabled = st.checkbox("Enable scheduling", value=bool(schedule_cfg.get("enabled", True)))
    times_text = st.text_area("Fixed times (one per line, HH:MM)", value="\n".join(schedule_cfg.get("times", [])))
    random_cfg = schedule_cfg.setdefault("random", {})
    random_enabled = st.checkbox("Enable Random Schedule", value=bool(random_cfg.get("enabled", False)))
    per_window = st.number_input("Times per window", min_value=1, max_value=5, value=int(random_cfg.get("per_window_count", 1)))
    regen_at = st.text_input("Regenerate daily at (HH:MM)", value=random_cfg.get("regenerate_at", "00:05"))
    windows_text = st.text_area(
        "Random windows (one per line: START-END, e.g. 08:00-11:00)",
        value="\n".join([f"{w.get('start','')}-{w.get('end','')}" for w in random_cfg.get("windows", [])])
    )

    if st.button("Save Schedule"):
        schedule_cfg["enabled"] = enabled
        schedule_cfg["times"] = [t.strip() for t in times_text.splitlines() if t.strip()]
        random_cfg["enabled"] = random_enabled
        random_cfg["per_window_count"] = int(per_window)
        random_cfg["regenerate_at"] = regen_at.strip()
        windows = []
        for line in windows_text.splitlines():
            if "-" in line:
                start, end = [s.strip() for s in line.split("-", 1)]
                windows.append({"start": start, "end": end})
        random_cfg["windows"] = windows
        save_config(config)
        st.success("Schedule saved")

    # Auto-refresh the schedule timer every second (avoid refresh during long runs)
    if not st.session_state.get("is_busy", False):
        st_autorefresh(interval=1000, key="schedule_timer_refresh")

    state = read_schedule_state()
    if state and state.get("times"):
        next_run = get_next_run_from_times(state["times"])
        if next_run:
            delta = next_run - datetime.now()
            st.info(f"Next scheduled run: {next_run.strftime('%Y-%m-%d %H:%M')} (in {str(delta).split('.')[0]})")
    else:
        # Fallback to fixed times from config
        fixed_times = schedule_cfg.get("times", [])
        next_run = get_next_run_from_times(fixed_times)
        if next_run:
            delta = next_run - datetime.now()
            st.info(f"Next scheduled run: {next_run.strftime('%Y-%m-%d %H:%M')} (in {str(delta).split('.')[0]})")

with st.expander("Advanced Config (YAML)", expanded=False):
    yaml_text = st.text_area("config.yaml", value=yaml.safe_dump(config, sort_keys=False), height=300)
    if st.button("Save YAML"):
        try:
            updated = yaml.safe_load(yaml_text) or {}
            save_config(updated)
            st.success("Config updated")
        except Exception as e:
            st.error(f"Invalid YAML: {e}")

st.divider()

st.header("🚀 Run / Test")
st.markdown("These actions run commands in the project folder.")
if debug_mode:
    st.caption("Debug mode is ON — full logs will show below each action.")

col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    if st.button("Authenticate YouTube"):
        st.write("Running: python main.py --auth")
        run_command_stream("Authenticate YouTube", ["python", "main.py", "--auth"], debug_mode, "logs_auth")
with col_b:
    if st.button("Run Once"):
        st.write("Running: python main.py --once")
        run_command_stream("Run Once", ["python", "main.py", "--once"], debug_mode, "logs_run")
with col_c:
    if st.button("Run Tests"):
        st.write("Running: python main.py --test")
        run_command_stream("Run Tests", ["python", "main.py", "--test"], debug_mode, "logs_test")
with col_d:
    run_count = st.number_input("Run X times", min_value=1, max_value=20, value=3)
    if st.button("Run X"):
        st.write(f"Running: python main.py --run {int(run_count)}")
        run_command_stream("Run X", ["python", "main.py", "--run", str(int(run_count))], debug_mode, "logs_run")

st.divider()
st.header("🎮 Background Videos")
st.markdown("Auto-search, download, and split gameplay clips.")
if debug_mode:
    st.caption("Debug mode is ON — full logs will show below each action.")

col_b1, col_b2, col_b3 = st.columns(3)
with col_b1:
    num_videos = st.number_input("Videos per style", min_value=1, max_value=10, value=3)
with col_b2:
    clips_per_video = st.number_input("Clips per video", min_value=1, max_value=20, value=5)
with col_b3:
    clip_duration = st.number_input("Clip duration (sec)", min_value=15, max_value=180, value=90)

if st.button("Download Gameplay Pack (Minecraft + Subway Surfers)"):
    st.write("Running: python background_generator.py --download-all")
    run_command_stream(
        "Download Backgrounds",
        [
            "python",
            "background_generator.py",
            "--download-all",
            "--num-videos",
            str(int(num_videos)),
            "--clips-per-video",
            str(int(clips_per_video)),
            "--clip-duration",
            str(int(clip_duration)),
        ],
        debug_mode,
        "logs_bg"
    )