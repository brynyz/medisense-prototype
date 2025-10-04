from rest_framework import serializers
from .models import Patient, SymptomLog


class PatientSerializer(serializers.ModelSerializer):
    class Meta:
        model = Patient
        fields = ('id', 'name', 'age', 'sex', 'course', 'date_logged')
        read_only_fields = ('id',)

    def validate_age(self, value):
        if value < 0 or value > 150:
            raise serializers.ValidationError("Age must be between 0 and 150")
        return value


class SymptomLogSerializer(serializers.ModelSerializer):
    patient_name = serializers.CharField(source='patient.name', read_only=True)
    patient = PatientSerializer(read_only=True)

    class Meta:
        model = SymptomLog
        fields = ('id', 'patient', 'patient_name', 'symptom', 'notes', 'date_logged')
        read_only_fields = ('id', 'patient_name')


class SymptomLogCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = SymptomLog
        fields = ('patient', 'symptom', 'notes', 'date_logged')


class PatientWithSymptomsSerializer(serializers.ModelSerializer):
    symptoms = SymptomLogSerializer(source='symptomlog_set', many=True, read_only=True)
    
    class Meta:
        model = Patient
        fields = ('id', 'name', 'age', 'sex', 'course', 'date_logged', 'symptoms')
        read_only_fields = ('id',)
