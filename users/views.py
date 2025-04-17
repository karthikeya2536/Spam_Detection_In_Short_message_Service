from django.shortcuts import render, redirect
from django.contrib import messages

from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)

        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return redirect('UserHome')
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})


def logout(request):
    try:
        del request.session['id']
        del request.session['loggeduser']
        del request.session['loginid']
        del request.session['email']
    except:
        pass
    return redirect('index')

# DatasetView function removed as it's not used in any URL pattern

import pandas as pd
import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Initialize global variables
tfidf_vectorizer = None
nb_classifier = None
X_test = None
y_test = None
X_test_tfidf = None

# Setup logging
logger = logging.getLogger(__name__)

def load_model():
    global tfidf_vectorizer, nb_classifier, X_test, y_test, X_test_tfidf

    try:
        # Construct the path to the dataset
        path = os.path.join(settings.MEDIA_ROOT, 'balanced_spam_dataset.csv')
        logger.info(f"Loading dataset from: {path}")

        # Check if file exists
        if not os.path.exists(path):
            logger.error(f"Dataset file not found at: {path}")
            return False

        # Load the dataset
        df = pd.read_csv(path)
        value_counts = df['Category'].value_counts()
        logger.info(f"Dataset loaded successfully. Categories: {value_counts}")

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(df['Message'], df['Category'], test_size=0.2, random_state=42)

        # Vectorize the messages using TF-IDF
        tfidf_vectorizer = TfidfVectorizer(max_features=10000)
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)

        # Train a Naive Bayes classifier
        nb_classifier = MultinomialNB()
        nb_classifier.fit(X_train_tfidf, y_train)
        logger.info("Model trained successfully")

        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

# Load the model when the module is imported
model_loaded = load_model()

def machine_learning(request):
    if not model_loaded or nb_classifier is None or X_test_tfidf is None or y_test is None:
        # Try to load the model again
        if not load_model():
            error_message = "Unable to load the machine learning model. Please contact the administrator."
            logger.error(error_message)
            return render(request, "users/machine_learning.html", {'error': error_message})

    try:
        predictions = nb_classifier.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, predictions)
        logger.info(f"Accuracy: {accuracy:.2f}")
        nb = classification_report(y_test, predictions, output_dict=True)
        nb = pd.DataFrame(nb).transpose()
        nb = pd.DataFrame(nb)
        return render(request, "users/machine_learning.html", {'acc': accuracy})
    except Exception as e:
        error_message = f"Error generating machine learning analysis: {str(e)}"
        logger.error(error_message)
        return render(request, "users/machine_learning.html", {'error': error_message})

'''
a classification label, with possible values including
spam (0), Not Spam (1).
'''


def prediction(request):
    # Check if model is loaded
    if not model_loaded or tfidf_vectorizer is None or nb_classifier is None:
        # Try to load the model again
        if not load_model():
            error_message = "Unable to load the prediction model. Please contact the administrator."
            logger.error(error_message)
            return render(request, 'users/predictForm.html', {'error': error_message})

    if request.method == 'POST':
        try:
            # Get the text from the form
            single_tweet = request.POST.get('tweets')
            if not single_tweet or single_tweet.strip() == '':
                return render(request, 'users/predictForm.html', {'error': 'Please enter some text to analyze'})

            logger.info(f"Analyzing text: {single_tweet[:50]}...")

            # Transform the text using the vectorizer
            single_tweet_tfidf = tfidf_vectorizer.transform([single_tweet])
            logger.info(f"Text vectorized successfully")

            # Make prediction
            single_prediction = nb_classifier.predict(single_tweet_tfidf)
            logger.info(f"Prediction result: {single_prediction[0]}")

            # Convert numerical prediction to text
            if single_prediction[0] == 0:
                single_prediction = 'spam'
            elif single_prediction[0] == 1:
                single_prediction = 'non-spam'
            else:
                single_prediction = f'unknown-{single_prediction[0]}'

            logger.info(f"Final prediction: {single_prediction}")

            # Return the result
            return render(request, 'users/predictForm.html', {
                'output': single_prediction,
                'input_text': single_tweet
            })

        except Exception as e:
            error_message = f"Error making prediction: {str(e)}"
            logger.error(error_message)
            return render(request, 'users/predictForm.html', {'error': error_message})

    return render(request, 'users/predictForm.html', {})