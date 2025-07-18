from django import forms
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.models import User
from captcha.fields import CaptchaField, CaptchaTextInput

class LoginForm(AuthenticationForm):
    captcha = CaptchaField(widget=CaptchaTextInput(attrs={'placeholder': 'Captcha'}))

class RegistrationForm(UserCreationForm):
    email = forms.EmailField(required=True)
    captcha = CaptchaField(widget=CaptchaTextInput(attrs={'placeholder': 'Captcha'}))

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']