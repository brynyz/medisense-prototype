from django.shortcuts import render
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import render
from .forms import UserCredentialsForm

# Create your views here.

def home(request):
    return render(request, 'accounts/homepage.html')

def login(request):
    if request.method == 'POST':
        form = UserCredentialsForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return render(request, 'accounts/home.html', {'user': user})
            else:
                return render(request, 'accounts/login.html', {'form': form, 'error': 'Invalid credentials'})
    else:
        form = UserCredentialsForm()
    return render(request, 'accounts/login.html', {'form': form})

def register(request):
    form = UserCredentialsForm(request.POST)
    return render(request, "accounts/register.html", {'form': form})
