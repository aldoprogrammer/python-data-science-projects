#!/bin/bash
echo "Setting up virtual environment for Gender Classification ML project..."
echo

echo "Creating virtual environment..."
python3 -m venv .venv

echo
echo "Activating virtual environment..."
source .venv/bin/activate

echo
echo "Installing required packages..."
pip install --upgrade pip
pip install -r requirements.txt

echo
echo "Setup complete!"
echo
echo "To activate the virtual environment in future sessions, run:"
echo "source .venv/bin/activate"
echo
echo "To run the project:"
echo "python app.py"
