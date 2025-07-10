from django.db import models

# Create your models here.

from django.db import models

class Patient(models.Model):
    name = models.CharField(max_length=100)
    age = models.IntegerField()
    sex = models.CharField(max_length=10)
    date_logged = models.DateField(auto_now_add=True)

    def __str__(self):
        return self.name

class SymptomLog(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
    symptom = models.CharField(max_length=255)
    notes = models.TextField(blank=True)

    def __str__(self):
        return f"{self.patient.name} - {self.symptom}"
