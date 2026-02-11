#!/bin/bash
# scripts/setup_env.sh - Setup development environment for RetinaViT

# Detect OS
OS_TYPE="$(uname)"
case "$OS_TYPE" in
    Linux*|Darwin*)
        ACTIVATE_CMD="source .venv/bin/activate"
        ;;
    MINGW*|CYGWIN*|MSYS*)
        ACTIVATE_CMD=".venv\\Scripts\\activate"
        ;;
    *)
        ACTIVATE_CMD="source .venv/bin/activate"
        ;;
esac

echo "--- Initializing Virtual Environment ---"
python -m venv .venv

echo "--- Installing Pinned Dependencies ---"
if [[ "$OS_TYPE" == *"MINGW"* || "$OS_TYPE" == *"MSYS"* ]]; then
    .venv/Scripts/python -m pip install --upgrade pip
    .venv/Scripts/pip install -r requirements.txt
else
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
fi

echo "--- Installing Pre-commit Hooks ---"
if [[ "$OS_TYPE" == *"MINGW"* || "$OS_TYPE" == *"MSYS"* ]]; then
    .venv/Scripts/pre-commit install
else
    .venv/bin/pre-commit install
fi

echo ""
echo "Setup complete!"
echo "To activate the environment, run:"
echo "  $ACTIVATE_CMD"
echo ""
