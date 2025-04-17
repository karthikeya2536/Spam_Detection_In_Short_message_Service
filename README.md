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

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Use the following settings:
   - Build Command: `pip install -r requirements.txt --no-cache-dir && python manage.py collectstatic --noinput && python manage.py migrate`
   - Start Command: `gunicorn Spam_Detection_In_Short_Message_Service.wsgi:application`
4. Add the following environment variables:
   - `SECRET_KEY`: Generate a secure secret key (you can use [Djecrety](https://djecrety.ir/))
   - `PYTHON_VERSION`: 3.11.2

### Important Files for Deployment

- **requirements.txt**: Lists all dependencies including gunicorn and whitenoise
- **Procfile**: Contains the command to run the application (`web: gunicorn Spam_Detection_In_Short_Message_Service.wsgi`)
- **runtime.txt**: Specifies the Python version (`python-3.11.2`)
- **settings.py**: Configured with:
  - `DEBUG = False` for production
  - `ALLOWED_HOSTS = ['*']` to accept connections from any host
  - Security settings for HTTPS
  - Static files configuration with WhiteNoise

### Post-Deployment Steps

1. After deployment, create a superuser to access the admin panel:
   ```
   python manage.py createsuperuser
   ```
2. Access the admin panel at `/admin` to manage users and data

## License

This project is licensed under the MIT License - see the LICENSE file for details.

