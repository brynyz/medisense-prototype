#!/usr/bin/env bash
set -o errexit

# Debug: Show current directory and list files
echo "Current directory: $(pwd)"
echo "Files in current directory:"
ls -la

# Check if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Found requirements.txt"
else
    echo "ERROR: requirements.txt not found in $(pwd)"
    exit 1
fi

# Install dependencies
pip install -r requirements.txt

# Navigate to Django project directory
cd medisense

# Show Django project directory contents
echo "Django project directory: $(pwd)"
ls -la

# Collect static files
python manage.py collectstatic --no-input

# Run migrations
python manage.py migrate