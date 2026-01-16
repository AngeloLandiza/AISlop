#!/usr/bin/env python3
"""
Streamlit UI for YouTube Shorts Automation.
"""

import json
import subprocess
import os
import platform
from pathlib import Path
from typing import Dict, Any

import streamlit as st
import yaml


PROJECT_DIR = Path(__file__).parent
CONFIG_PATH = PROJECT_DIR / "config.yaml"
CREDENTIALS_DIR = PROJECT_DIR / "credentials"
CLIENT_SECRET_PATH = CREDENTIALS_DIR / "client_secret.json"

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

col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    if st.button("Authenticate YouTube"):
        st.write("Running: python main.py --auth")
        subprocess.run(["python", "main.py", "--auth"], cwd=str(PROJECT_DIR))
with col_b:
    if st.button("Run Once"):
        st.write("Running: python main.py --once")
        subprocess.run(["python", "main.py", "--once"], cwd=str(PROJECT_DIR))
with col_c:
    if st.button("Run Tests"):
        st.write("Running: python main.py --test")
        subprocess.run(["python", "main.py", "--test"], cwd=str(PROJECT_DIR))
with col_d:
    run_count = st.number_input("Run X times", min_value=1, max_value=20, value=3)
    if st.button("Run X"):
        st.write(f"Running: python main.py --run {int(run_count)}")
        subprocess.run(["python", "main.py", "--run", str(int(run_count))], cwd=str(PROJECT_DIR))