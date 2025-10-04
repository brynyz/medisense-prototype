import csv
import os
from datetime import datetime
from django.core.management.base import BaseCommand
from django.conf import settings
from patients.models import Patient, SymptomLog


class Command(BaseCommand):
    help = 'Import patient and symptom data from cleaned CSV file'

    def add_arguments(self, parser):
        parser.add_argument(
            '--file',
            type=str,
            default='data/cleaned/finalcleaned.csv',
            help='Path to CSV file relative to project root'
        )
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear existing data before import'
        )

    def handle(self, *args, **options):
        # Get the CSV file path
        csv_file = options['file']
        if not os.path.isabs(csv_file):
            csv_file = os.path.join(settings.BASE_DIR, csv_file)

        if not os.path.exists(csv_file):
            self.stdout.write(
                self.style.ERROR(f'CSV file not found: {csv_file}')
            )
            return

        # Clear existing data if requested
        if options['clear']:
            self.stdout.write('Clearing existing data...')
            SymptomLog.objects.all().delete()
            Patient.objects.all().delete()
            
            # Reset auto-increment IDs to 1
            from django.db import connection
            with connection.cursor() as cursor:
                # Reset Patient table ID sequence (MySQL)
                cursor.execute("ALTER TABLE patients_patient AUTO_INCREMENT = 1;")
                # Reset SymptomLog table ID sequence (MySQL)
                cursor.execute("ALTER TABLE patients_symptomlog AUTO_INCREMENT = 1;")
            
            self.stdout.write(self.style.SUCCESS('Existing data cleared and IDs reset to 1.'))

        # Import data
        self.stdout.write(f'Importing data from {csv_file}...')
        
        imported_count = 0
        skipped_count = 0
        
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for row in reader:
                try:
                    # Skip rows with missing essential data
                    if not row['date_cleaned'] or not row['course_mapped']:
                        skipped_count += 1
                        continue
                    
                    # Parse date
                    visit_date = datetime.strptime(row['date_cleaned'], '%Y-%m-%d').date()
                    
                    # Handle missing/empty values
                    age = None
                    if row['age'] and row['age'].strip() and row['age'] != 'nan':
                        try:
                            age = int(float(row['age']))
                        except (ValueError, TypeError):
                            age = None
                    
                    gender = row['gender'] if row['gender'] and row['gender'].strip() else 'Unknown'
                    course = row['course_mapped'].lower() if row['course_mapped'] else 'unknown'
                    symptom = row['normalized_symptoms'] if row['normalized_symptoms'] and row['normalized_symptoms'].strip() else 'unspecified'
                    
                    # Create a unique patient for each row (each row represents a different patient visit)
                    # Generate unique patient identifier using row number or timestamp
                    patient_identifier = f"Patient_{imported_count + 1:04d}"
                    
                    # Create a new patient for each row since each row represents a different patient
                    patient = Patient.objects.create(
                        name=patient_identifier,
                        age=age or 20,
                        course=course,
                        sex=gender,
                        date_logged=visit_date
                    )
                    
                    # Create symptom log
                    SymptomLog.objects.create(
                        patient=patient,
                        symptom=symptom,
                        notes='Imported from CSV',
                        date_logged=visit_date
                    )
                    
                    imported_count += 1
                    
                    if imported_count % 50 == 0:
                        self.stdout.write(f'Imported {imported_count} records...')
                        
                except Exception as e:
                    self.stdout.write(
                        self.style.WARNING(f'Error importing row: {row}. Error: {str(e)}')
                    )
                    skipped_count += 1
                    continue

        self.stdout.write(
            self.style.SUCCESS(
                f'Import completed! Imported: {imported_count}, Skipped: {skipped_count}'
            )
        )
        
        # Show summary statistics
        total_patients = Patient.objects.count()
        total_symptoms = SymptomLog.objects.count()
        
        self.stdout.write(f'Total patients in database: {total_patients}')
        self.stdout.write(f'Total symptom logs in database: {total_symptoms}')
