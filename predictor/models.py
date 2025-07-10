from django.db import models
from patients.models import Patient

class Prediction(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
    predicted_medicine = models.CharField(max_length=100)
    confidence_score = models.FloatField()
    date_predicted = models.DateField(auto_now_add=True)

    def __str__(self):
        return f"Prediction for {self.patient.name}"
