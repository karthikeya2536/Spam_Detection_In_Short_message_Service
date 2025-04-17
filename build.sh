#!/usr/bin/env bash
# exit on error
set -o errexit

# Install dependencies
pip install -r requirements.txt

# Collect static files
python manage.py collectstatic --no-input

# Apply database migrations
python manage.py migrate

# Create media directory if it doesn't exist
mkdir -p media

# Make sure the dataset file is in the media directory
if [ ! -f "media/balanced_spam_dataset.csv" ]; then
    echo "Copying dataset file to media directory..."
    cp -f balanced_spam_dataset.csv media/ || echo "Warning: Could not copy dataset file"
fi

# Print debug information
echo "Media directory contents:"
ls -la media/
