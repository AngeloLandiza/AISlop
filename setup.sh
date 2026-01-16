#!/bin/bash
# =============================================================================
# YouTube Shorts Automation - Setup Script
# =============================================================================
# This script sets up your development environment.
#
# Usage: ./setup.sh
# =============================================================================

set -e  # Exit on error

echo ""
echo "=============================================="
echo "  YouTube Shorts Automation Setup"
echo "=============================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# -----------------------------------------------------------------------------
# Step 1: Check Python
# -----------------------------------------------------------------------------
echo "1. Checking Python..."

if command -v python3 &> /dev/null; then
    PYTHON=python3
    PIP=pip3
elif command -v python &> /dev/null; then
    PYTHON=python
    PIP=pip
else
    echo "   ✗ Python not found!"
    echo "   Please install Python 3.8+ from https://python.org"
    exit 1
fi

PYTHON_VERSION=$($PYTHON --version 2>&1 | cut -d' ' -f2)
echo "   ✓ Python $PYTHON_VERSION found"

# -----------------------------------------------------------------------------
# Step 2: Create Virtual Environment
# -----------------------------------------------------------------------------
echo ""
echo "2. Creating virtual environment..."

if [ ! -d "venv" ]; then
    $PYTHON -m venv venv
    echo "   ✓ Virtual environment created"
else
    echo "   ✓ Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# -----------------------------------------------------------------------------
# Step 3: Install Dependencies
# -----------------------------------------------------------------------------
echo ""
echo "3. Installing Python dependencies..."

pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "   ✓ Dependencies installed"

# -----------------------------------------------------------------------------
# Step 4: Check FFmpeg
# -----------------------------------------------------------------------------
echo ""
echo "4. Checking FFmpeg..."

if command -v ffmpeg &> /dev/null; then
    FFMPEG_VERSION=$(ffmpeg -version 2>&1 | head -1)
    echo "   ✓ $FFMPEG_VERSION"
else
    echo "   ⚠ FFmpeg not found!"
    echo ""
    echo "   Please install FFmpeg:"
    echo "   - macOS:   brew install ffmpeg"
    echo "   - Ubuntu:  sudo apt-get install ffmpeg"
    echo "   - Windows: Download from https://ffmpeg.org/download.html"
    echo ""
fi

# -----------------------------------------------------------------------------
# Step 5: Create Required Directories
# -----------------------------------------------------------------------------
echo ""
echo "5. Creating directories..."

mkdir -p videos logs output
echo "   ✓ videos/ logs/ output/ directories created"

# -----------------------------------------------------------------------------
# Step 6: Check Configuration Files
# -----------------------------------------------------------------------------
echo ""
echo "6. Checking configuration..."

if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "   Creating .env from template..."
        cp .env.example .env
        echo "   ✓ .env created (edit with your API keys)"
    fi
fi

# Check for required credential files
missing_files=false

if [ ! -f "client_secret.json" ]; then
    echo "   ⚠ client_secret.json not found (needed for YouTube)"
    missing_files=true
fi

if [ ! -f "google-credentials.json" ]; then
    echo "   ⚠ google-credentials.json not found (needed for TTS)"
    missing_files=true
fi

# Check config.yaml API key
if grep -q "YOUR_COHERE_API_KEY_HERE" config.yaml 2>/dev/null; then
    echo "   ⚠ Cohere API key not configured in config.yaml"
    missing_files=true
fi

if [ "$missing_files" = false ]; then
    echo "   ✓ All configuration files present"
fi

# -----------------------------------------------------------------------------
# Step 7: Summary
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""

if [ "$missing_files" = true ]; then
    echo "⚠  Some configuration is missing. Please:"
    echo ""
    echo "1. Get Cohere API key:"
    echo "   → https://dashboard.cohere.io/"
    echo "   → Add to config.yaml"
    echo ""
    echo "2. Get Google Cloud credentials:"
    echo "   → https://console.cloud.google.com/"
    echo "   → Enable 'YouTube Data API v3' and 'Cloud Text-to-Speech API'"
    echo "   → Create OAuth 2.0 credentials → Download as 'client_secret.json'"
    echo "   → Create Service Account → Download as 'google-credentials.json'"
    echo ""
    echo "3. Add background videos to the 'videos/' directory"
    echo "   (MP4 format, any action/parkour style gameplay)"
    echo ""
fi

echo "Quick Start Commands:"
echo "---------------------"
echo ""
echo "  # Activate virtual environment"
echo "  source venv/bin/activate"
echo ""
echo "  # Authenticate with YouTube (first time)"
echo "  python main.py --auth"
echo ""
echo "  # Test all components"
echo "  python main.py --test"
echo ""
echo "  # Generate and upload one video"
echo "  python main.py --once"
echo ""
echo "  # Start automated scheduling"
echo "  python main.py --schedule"
echo ""
echo "=============================================="
echo ""
