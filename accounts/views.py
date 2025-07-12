from django.shortcuts import render, redirect #render and redirect are used to render templates and redirect users
from django.contrib.auth.views import LoginView #default login view from django
from django.contrib.auth.decorators import login_required #login required decorator to restrict access to certain views
from django.contrib import messages #display messages if needed
from django.shortcuts import render #render templetaes
from django.urls import reverse_lazy #reverser url patters

from .forms import LoginForm, RegistrationForm

# Create your views here.

class CustomLoginView(LoginView):
    authentication_form = LoginForm
    redirect_authenticated_user = True
    template_name = 'accounts/account.html'
    success_url = reverse_lazy('home')

    def get_success_url(self):
        return self.success_url

@login_required
def home(request):
    return render(request, 'accounts/homepage.html')

@login_required
def logout_view(request):
    request.session.flush()
    return redirect('login')

@login_required
def register(request):
    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.is_active = False  #admin approval in django admin page
            user.save()

            return redirect('login') 
    else:
        form = RegistrationForm()
    return render(request, 'accounts/account.html', {'form': form, 'mode': 'register'})