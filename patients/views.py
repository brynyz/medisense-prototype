from django.shortcuts import render, redirect, get_object_or_404
from .models import SymptomLog
from django.views.decorators.csrf import csrf_exempt

def symptom_log_view(request):
    symptom_logs = SymptomLog.objects.all().order_by('-date_logged')
    return render(request, 'symptoms/symptom_log.html', {
        'symptom_logs': symptom_logs,
        'gender_choices': SymptomLog.GENDER_CHOICES,
    })

@csrf_exempt
def add_symptom(request):
    if request.method == 'POST':
        SymptomLog.objects.create(
            age=request.POST.get('age'),
            gender=request.POST.get('gender'),
            symptom=request.POST.get('symptom'),
        )
        return redirect('symptom_log')

    return redirect('symptom_log')


def edit_symptom(request, log_id):
    log = get_object_or_404(SymptomLog, id=log_id)

    if request.method == 'POST':
        log.age = request.POST.get('age')
        log.gender = request.POST.get('gender')
        log.symptom = request.POST.get('symptom')
        log.save()
        return redirect('symptom_log')

    symptom_logs = SymptomLog.objects.all().order_by('-date_logged')
    return render(request, 'symptoms/symptom_log.html', {
        'symptom_logs': symptom_logs,
        'edit_log': log,
        'gender_choices': SymptomLog.GENDER_CHOICES,
    })


def delete_symptom(request, log_id):
    log = get_object_or_404(SymptomLog, id=log_id)
    log.delete()
    return redirect('symptom_log')
