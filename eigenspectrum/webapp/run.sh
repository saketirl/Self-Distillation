#!/bin/bash
# Run the eigenspectrum viewer web application

cd "$(dirname "$0")"

# Check if flask is installed
if ! python -c "import flask" 2>/dev/null; then
    echo "Installing Flask..."
    pip install flask numpy
fi

echo "Starting Eigenspectrum Viewer..."
echo "Open http://localhost:5000 in your browser"
echo ""

python app.py
