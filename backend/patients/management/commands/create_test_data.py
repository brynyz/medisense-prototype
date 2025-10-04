from django.core.management.base import BaseCommand
from django.utils import timezone
from patients.models import Patient, SymptomLog
from datetime import datetime, timedelta
import random

class Command(BaseCommand):
    help = 'Create test data for patients and symptoms'

    def handle(self, *args, **options):
        # Clear existing data
        SymptomLog.objects.all().delete()
        Patient.objects.all().delete()
        
        # Sample data
        courses = ['Computer Science', 'Nursing', 'Medicine', 'Engineering', 'Business']
        symptoms = [
            'Headache', 'Fever', 'Cough', 'Sore throat', 'Nausea', 
            'Stomach pain', 'Dizziness', 'Fatigue', 'Body aches', 'Runny nose'
        ]
        names = [
            'John Doe', 'Jane Smith', 'Mike Johnson', 'Sarah Wilson', 'David Brown',
            'Lisa Garcia', 'Tom Miller', 'Anna Davis', 'Chris Lee', 'Maria Rodriguez'
        ]
        
        # Create patients
        patients = []
        for i in range(20):
            patient = Patient.objects.create(
                name=names[i % len(names)] + f" {i+1}",
                age=random.randint(18, 25),
                sex=random.choice(['Male', 'Female']),
                course=random.choice(courses),
                date_logged=timezone.now() - timedelta(days=random.randint(0, 30))
            )
            patients.append(patient)
            self.stdout.write(f"Created patient: {patient.name}")
        
        # Create symptom logs
        for i in range(50):
            patient = random.choice(patients)
            symptom = random.choice(symptoms)
            notes = f"Patient reported {symptom.lower()} during visit"
            
            SymptomLog.objects.create(
                patient=patient,
                symptom=symptom,
                notes=notes
            )
            self.stdout.write(f"Created symptom log: {symptom} for {patient.name}")
        
        self.stdout.write(
            self.style.SUCCESS(
                f'Successfully created {len(patients)} patients and 50 symptom logs'
            )
        )
