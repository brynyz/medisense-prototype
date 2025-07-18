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
            user.is_active=True #admin approval in django admin page
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

    # Get database engine
    db_engine = settings.DATABASES['default']['ENGINE']
    
    try:
        if 'mysql' in db_engine:
            # MySQL backup using mysqldump
            command = [
                'mysqldump',
                '-u', settings.DATABASES['default']['USER'],
                f"-p{settings.DATABASES['default']['PASSWORD']}",
                '--host', settings.DATABASES['default'].get('HOST', 'localhost'),
                '--port', str(settings.DATABASES['default'].get('PORT', 3306)),
                settings.DATABASES['default']['NAME']
            ]
        
        elif 'postgresql' in db_engine:
            # PostgreSQL backup using pg_dump
            command = [
                'pg_dump',
                '--host', settings.DATABASES['default'].get('HOST', 'localhost'),
                '--port', str(settings.DATABASES['default'].get('PORT', 5432)),
                '--username', settings.DATABASES['default']['USER'],
                '--dbname', settings.DATABASES['default']['NAME'],
                '--no-password',  # Will use PGPASSWORD environment variable
                '--verbose',
                '--clean',
                '--no-owner',
                '--no-privileges'
            ]
            
            # Set password environment variable for PostgreSQL
            env = os.environ.copy()
            env['PGPASSWORD'] = settings.DATABASES['default']['PASSWORD']
        
        elif 'sqlite3' in db_engine:
            # SQLite backup (simple file copy)
            import shutil
            db_path = settings.DATABASES['default']['NAME']
            shutil.copy2(db_path, filepath.replace('.sql', '.sqlite'))
            
            with open(filepath.replace('.sql', '.sqlite'), 'rb') as sqlite_file:
                response = HttpResponse(sqlite_file.read(), content_type='application/x-sqlite3')
                response['Content-Disposition'] = f'attachment; filename="{filename.replace(".sql", ".sqlite")}"'
                return response
        
        else:
            messages.error(request, f'Unsupported database engine: {db_engine}')
            return redirect('settings')

        # Execute backup command for MySQL/PostgreSQL
        with open(filepath, 'w') as out:
            if 'postgresql' in db_engine:
                result = subprocess.run(command, stdout=out, stderr=subprocess.PIPE, env=env, text=True)
            else:
                result = subprocess.run(command, stdout=out, stderr=subprocess.PIPE, text=True)
        
        # Check if backup was successful
        if result.returncode != 0:
            messages.error(request, f'Backup failed: {result.stderr}')
            return redirect('settings')

        # Return the backup file as download
        with open(filepath, 'rb') as sql_file:
            response = HttpResponse(sql_file.read(), content_type='application/sql')
            response['Content-Disposition'] = f'attachment; filename="{filename}"'
            return response
            
    except FileNotFoundError as e:
        messages.error(request, f'Database backup tool not found. Please install the required database client tools.')
        return redirect('settings')
    except Exception as e:
        messages.error(request, f'Backup failed: {str(e)}')
        return redirect('settings')

@login_required
def restore_database(request):
    if request.method == 'POST' and request.FILES.get('sql_file'):
        sql_file = request.FILES['sql_file']
        filepath = os.path.join(settings.BASE_DIR, 'backups', sql_file.name)

        # Save uploaded file
        with open(filepath, 'wb+') as destination:
            for chunk in sql_file.chunks():
                destination.write(chunk)

        # Get database engine
        db_engine = settings.DATABASES['default']['ENGINE']
        
        try:
            if 'mysql' in db_engine:
                # MySQL restore
                command = [
                    'mysql',
                    '-u', settings.DATABASES['default']['USER'],
                    f"-p{settings.DATABASES['default']['PASSWORD']}",
                    '--host', settings.DATABASES['default'].get('HOST', 'localhost'),
                    '--port', str(settings.DATABASES['default'].get('PORT', 3306)),
                    settings.DATABASES['default']['NAME']
                ]
                
                with open(filepath, 'r') as input_file:
                    result = subprocess.run(command, stdin=input_file, stderr=subprocess.PIPE, text=True)
            
            elif 'postgresql' in db_engine:
                # PostgreSQL restore using psql
                command = [
                    'psql',
                    '--host', settings.DATABASES['default'].get('HOST', 'localhost'),
                    '--port', str(settings.DATABASES['default'].get('PORT', 5432)),
                    '--username', settings.DATABASES['default']['USER'],
                    '--dbname', settings.DATABASES['default']['NAME'],
                    '--no-password',
                    '--file', filepath
                ]
                
                # Set password environment variable for PostgreSQL
                env = os.environ.copy()
                env['PGPASSWORD'] = settings.DATABASES['default']['PASSWORD']
                
                result = subprocess.run(command, stderr=subprocess.PIPE, env=env, text=True)
            
            elif 'sqlite3' in db_engine:
                # SQLite restore (replace database file)
                import shutil
                db_path = settings.DATABASES['default']['NAME']
                
                if sql_file.name.endswith('.sqlite'):
                    shutil.copy2(filepath, db_path)
                    result = type('Result', (), {'returncode': 0, 'stderr': ''})()
                else:
                    messages.error(request, 'Invalid file format for SQLite database.')
                    return redirect('settings')
            
            else:
                messages.error(request, f'Unsupported database engine: {db_engine}')
                return redirect('settings')

            # Check if restore was successful
            if result.returncode == 0:
                messages.success(request, 'Database restored successfully.')
            else:
                messages.error(request, f'Restore failed: {result.stderr}')
                
        except FileNotFoundError as e:
            messages.error(request, f'Database restore tool not found. Please install the required database client tools.')
        except Exception as e:
            messages.error(request, f'Restore failed: {str(e)}')

        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass

        return redirect('settings')

    return render(request, 'accounts/settings.html')

@login_required
def profile_settings(request):
    return render(request, 'accounts/profile_settings.html')