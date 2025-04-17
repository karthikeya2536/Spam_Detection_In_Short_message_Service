# Spam Detection in Short Message Service

A Django web application for detecting spam in text messages using machine learning.

## Features

- User registration and authentication
- Admin dashboard to manage users
- Machine learning model for spam detection
- Simple and intuitive UI

## Technology Stack

- Django
- Scikit-learn
- Pandas
- Bootstrap

## Deployment Instructions

### Prerequisites

- Python 3.11.2
- pip

### Local Development

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run migrations:
   ```
   python manage.py migrate
   ```
5. Start the development server:
   ```
   python manage.py runserver
   ```

### Deployment to Render
for live demo go through this link:
https://spam-detection-in-short-message-service.onrender.com



