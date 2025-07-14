#!/usr/bin/env bash
set -o errexit

# Navigate to Django project directory
cd medisense

# Install dependencies
pip install -r requirements.txt

# Collect static files
python manage.py collectstatic --no-input

# Run migrations
python manage.py migrate