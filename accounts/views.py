from django.shortcuts import render, redirect #render and redirect are used to render templates and redirect users
from django.contrib.auth.views import LoginView #default login view from django
from django.contrib.auth.decorators import login_required #login required decorator to restrict access to certain views
from django.contrib import messages #display messages if needed
from django.shortcuts import render #render templetaes
from django.urls import reverse_lazy #reverser url patters
from django.contrib.auth.models import User #import user model

import os
import subprocess
import json
from django.http import HttpResponse
from django.conf import settings
from django.utils.timezone import now

from .forms import LoginForm, RegistrationForm

from chartjs.views.lines import BaseLineChartView

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
        # Sample data for charts
    patient_trends = [65, 59, 80, 81, 56, 55, 40]
    inventory_data = [28, 48, 40, 19, 86, 27, 90]
    
    context = {
        'patient_trends': json.dumps(patient_trends),
        'inventory_data': json.dumps(inventory_data),
    }
    return render(request, 'accounts/homepage.html', context)

@login_required
def logout_view(request):
    request.session.flush()
    return redirect('login')

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

def settings_view(request):
    return render(request, 'accounts/settings.html')

@login_required
def backup_database(request):
    filename = f"backup_{now().strftime('%Y%m%d_%H%M%S')}.sql"
    filepath = os.path.join(settings.BASE_DIR, 'backups', filename)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    command = [
        'mysqldump',
        '-u', settings.DATABASES['default']['USER'],
        f"-p{settings.DATABASES['default']['PASSWORD']}",
        settings.DATABASES['default']['NAME']
    ]

    with open(filepath, 'w') as out:
        subprocess.run(command, stdout=out)

    with open(filepath, 'rb') as sql_file:
        response = HttpResponse(sql_file.read(), content_type='application/sql')
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response
    

def restore_database(request):
    if request.method == 'POST' and request.FILES.get('sql_file'):
        sql_file = request.FILES['sql_file']
        filepath = os.path.join(settings.BASE_DIR, 'backups', sql_file.name)

        with open(filepath, 'wb+') as destination:
            for chunk in sql_file.chunks():
                destination.write(chunk)

        command = [
            'mysql',
            '-u', settings.DATABASES['default']['USER'],
            f"-p{settings.DATABASES['default']['PASSWORD']}",
            settings.DATABASES['default']['NAME']
        ]

        with open(filepath, 'r') as input_file:
            subprocess.run(command, stdin=input_file)

        messages.success(request, 'Database restored successfully.')
        return redirect('settings')

    return render(request, 'accounts/settings.html')


def profile_settings(request):
    return render(request, 'accounts/profile_settings.html')