from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
from django.db.models import Count, Q
from django.db.models.functions import TruncDate
from django.utils import timezone
from datetime import datetime, timedelta
from .models import Patient, SymptomLog
from .serializers import (
    PatientSerializer, 
    SymptomLogSerializer, 
    SymptomLogCreateSerializer
)


class PatientViewSet(viewsets.ModelViewSet):
    queryset = Patient.objects.all()
    serializer_class = PatientSerializer
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]
    pagination_class = None  # Disable pagination for this viewset

    def get_serializer_class(self):
        if self.action == 'create':
            return PatientSerializer
        return PatientSerializer

    @action(detail=True, methods=['get'])
    def symptoms(self, request, pk=None):
        """Get all symptoms for a specific patient"""
        patient = self.get_object()
        symptoms = SymptomLog.objects.filter(patient=patient)
        serializer = SymptomLogSerializer(symptoms, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['get'])
    def statistics(self, request):
        """Get patient statistics"""
        total_patients = Patient.objects.count()
        patients_by_sex = Patient.objects.values('sex').annotate(count=Count('sex'))
        recent_patients = Patient.objects.order_by('-date_logged')[:10]
        
        stats = {
            'total_patients': total_patients,
            'patients_by_sex': list(patients_by_sex),
            'recent_patients': PatientSerializer(recent_patients, many=True).data
        }
        return Response(stats)


class SymptomLogViewSet(viewsets.ModelViewSet):
    queryset = SymptomLog.objects.all()
    serializer_class = SymptomLogSerializer
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]
    pagination_class = None  # Disable pagination for this viewset

    def get_serializer_class(self):
        if self.action == 'create':
            return SymptomLogCreateSerializer
        return SymptomLogSerializer

    def get_queryset(self):
        queryset = SymptomLog.objects.select_related('patient').all()
        patient_id = self.request.query_params.get('patient', None)
        symptom = self.request.query_params.get('symptom', None)
        
        if patient_id is not None:
            queryset = queryset.filter(patient_id=patient_id)
        if symptom is not None:
            queryset = queryset.filter(symptom__icontains=symptom)
            
        return queryset.order_by('-id')

    @action(detail=False, methods=['get'])
    def common_symptoms(self, request):
        """Get most common symptoms"""
        symptoms = SymptomLog.objects.values('symptom').annotate(
            count=Count('symptom')
        ).order_by('-count')[:10]
        return Response(symptoms)

    @action(detail=False, methods=['get'])
    def symptom_trends(self, request):
        """Get symptom trends over time"""
        trends = SymptomLog.objects.annotate(
            date=TruncDate('patient__date_logged')
        ).values('date').annotate(
            count=Count('id')
        ).order_by('date')
        
        return Response(list(trends))

    @action(detail=False, methods=['get'])
    def chart_data(self, request):
        """Get data formatted for time-series charts"""
        # Map symptoms to categories
        symptom_categories = {
            'respiratory': ['cough', 'fever', 'cold', 'flu', 'sore throat', 'runny nose', 'congestion'],
            'gastrointestinal': ['stomach ache', 'nausea', 'diarrhea', 'vomiting', 'indigestion'],
            'neurological': ['headache', 'migraine', 'dizziness', 'fatigue'],
            'musculoskeletal': ['back pain', 'joint pain', 'muscle pain', 'sprain', 'strain']
        }
        
        # Get symptom data grouped by week
        from django.db.models.functions import Extract
        from django.db.models import Case, When, CharField, Value
        
        # Categorize symptoms
        symptom_case = Case(
            *[When(symptom__icontains=symptom, then=Value(category)) 
              for category, symptoms in symptom_categories.items() 
              for symptom in symptoms],
            default=Value('other'),
            output_field=CharField()
        )
        
        # Group by week and category
        weekly_data = SymptomLog.objects.annotate(
            category=symptom_case,
            week=TruncDate('date_logged', kind='week')
        ).values('week', 'category').annotate(
            count=Count('id')
        ).order_by('week', 'category')
        
        # Format for frontend
        chart_data = {}
        for item in weekly_data:
            category = item['category'].title()
            if category not in chart_data:
                chart_data[category] = []
            
            chart_data[category].append({
                'x': item['week'].strftime('%Y-%m-%d'),
                'y': item['count']
            })
        
        # Convert to Nivo format
        formatted_data = []
        for category, data in chart_data.items():
            formatted_data.append({
                'id': category,
                'data': data
            })
        
        return Response(formatted_data)

    @action(detail=False, methods=['get'])
    def department_visits(self, request):
        """Get visit counts by department for campus heatmap"""
        # Map courses to departments
        department_mapping = {
            # Staff & Others
            "staff": "staff",
            "others": "oth",

            # College of Laws
            "bslm": "law",            # Bachelor of Science in Legal Management
            "jurisdoctor": "law",     # Juris Doctor (Law degree)

            # College of Business Management
            "bsba": "cbm",
            "bstm": "cbm",
            "bsentrep": "cbm",
            "bshm": "cbm",
            "bsma": "cbm",
            "bsais": "cbm",

            # College of Criminal Justice Education
            "bscrim": "ccje",
            "bslea": "ccje",          # Law Enforcement Administration (missing)

            # College of Education
            "bse": "ced",
            "beed": "ced",
            "bsed": "ced",
            "bped": "ced",
            "ced": "ced",
            "btved": "ced",           # Technical-Vocational Teacher Education (missing)
            "btle": "ced",            # Technology and Livelihood Education (missing)

            # College of Computing Studies, ICT
            "bscs": "ccsict",
            "bsit": "ccsict",
            "bsemc": "sas",           # Entertainment and Multi Media Computing (reclassified to SAS)
            "bsis": "ccsict",         # Information Systems (missing)
            "ccsict": "ccsict",

            # School of Arts & Sciences
            "baels": "sas",
            "bapos": "sas",
            "bacomm": "sas",          # Communication (missing)
            "bsbio": "sas",           # Biology (missing)
            "bsmath": "sas",          # Mathematics (missing)
            "bschem": "sas",          # Chemistry (missing)
            "bspsych": "sas",         # Psychology (missing)

            # Polytechnic School
            "bsitelectech": "poly",
            "bsitautotech": "poly",
            "bsindtech": "poly",      # Industrial Technology (missing)
            "mechanicaltech": "poly", # Mechanical Technology (missing)
            "refrigaircondtech": "poly", # Refrigeration & Airconditioning (missing)
            "assocaircraftmaint": "poly", # Aircraft Maintenance (missing)

            # Agriculture
            "bat": "agri",
            "bsagri": "agri",         # Agriculture major (missing)
            "bsagribiz": "agri",      # Agribusiness (missing)
            "bsenvi": "agri",         # Environmental Science (missing)
            "bsfor": "agri",          # Forestry (missing)
            "bsfisheries": "agri",    # Fisheries and Aquatic Sciences (missing)

            # Graduate Programs
            "dit": "grad",            # Doctor of Information Technology
            "mit": "grad",            # Master in Information Technology
            "ddsa": "grad",           # Diploma in Data Science Analytics
            "mba": "grad",            # Master of Business Administration (Extension)
            "mpa": "grad",            # Master in Public Administration (Extension)
            "masterlaws": "grad",     # Master of Laws (Consortium)
            "maed": "grad",           # MA in Education
            "phd_ed": "grad",         # PhD in Education
            "phd_animal": "grad",     # PhD in Animal Science
            "phd_crop": "grad",       # PhD in Crop Science
        }
        
        # Get department counts
        department_counts = {}
        
        # Group courses by department
        dept_groups = {}
        for course_code, dept_code in department_mapping.items():
            if dept_code not in dept_groups:
                dept_groups[dept_code] = []
            dept_groups[dept_code].append(course_code)
        
        # Count symptom visits (SymptomLog records) for each department
        for dept_code, course_codes in dept_groups.items():
            count = SymptomLog.objects.filter(
                Q(*[Q(patient__course__icontains=course) for course in course_codes], _connector=Q.OR)
            ).count()
            department_counts[dept_code.upper()] = count
        
        return Response(department_counts)

    @action(detail=False, methods=['get'])
    def symptom_trends(self, request):
        """Get symptom trends over time for line chart"""
        try:
            from django.db.models import Count, Min, Max
            from django.db.models.functions import TruncWeek
            from datetime import datetime, timedelta
            
            # Get date range (default to last 6 months)
            end_date = timezone.now().date()
            start_date = end_date - timedelta(days=180)
        
            # Allow custom date range via query params
            if request.GET.get('start_date'):
                start_date = datetime.strptime(request.GET.get('start_date'), '%Y-%m-%d').date()
            if request.GET.get('end_date'):
                end_date = datetime.strptime(request.GET.get('end_date'), '%Y-%m-%d').date()
        
            # Define symptom categories mapping (based on your ML pipeline categorization)
            symptom_categories = {
            'Respiratory': [
                'cold', 'cough', 'asthma', 'runny nose', 'sore throat', 'itchy throat', 
                'auri', 'shortness of breath', 'hyperventilation', 'earache', 'nosebleed',
                'hypertension'
            ],
            'Digestive': [
                'stomach ache', 'hyperacidity', 'lbm', 'diarrhea', 'vomiting', 
                'epigastric pain', 'dry mouth'
            ],
            'Pain & Musculoskeletal': [
                'headache', 'body pain', 'muscle strain', 'chest pain', 'toothache', 
                'dysmenorrhea', 'cramps', 'menstrual cramps', 'stiff neck', 'migraine',
                'sprain'
            ],
            'Dermatological & Trauma': [
                'abrasion', 'wound', 'punctured wound', 'cut', 'pimple', 
                'hematoma', 'stitches'
            ],
            'Neurological & Psychological': [
                'dizziness', 'anxiety'
            ],
            'Systemic & Infectious': [
                'fever', 'malaise', 'infection', 'uti', 'clammy skin', 'allergy', 
                'skin allergy'
            ]
            }
            
            # Get the earliest and latest dates from the database
            try:
                first_record = SymptomLog.objects.order_by('date_logged').first()
                last_record = SymptomLog.objects.order_by('-date_logged').first()
                
                if first_record and last_record:
                    actual_start = first_record.date_logged.date()
                    actual_end = last_record.date_logged.date()
                else:
                    actual_start = None
                    actual_end = None
                
            except Exception as date_error:
                actual_start = None
                actual_end = None
            
            if actual_start and actual_end:
                # Use only the actual data range (Sep 2022 to latest record)
                start_date = actual_start
                end_date = actual_end
            else:
                # Fallback if no data exists
                print("⚠️ No data found, using fallback date range")
                start_date = timezone.now().date() - timedelta(days=30)
                end_date = timezone.now().date()
            
            # Get symptom logs using the actual data range
            symptom_logs_raw = SymptomLog.objects.filter(
                date_logged__gte=start_date,
                date_logged__lte=end_date
            ).values('date_logged', 'symptom')
            
            # Debug: Show sample symptoms from database
            if len(symptom_logs_raw) > 0:
                sample_symptoms = [log['symptom'] for log in list(symptom_logs_raw)[:10]]
                print(f"🔍 Sample symptoms from database: {sample_symptoms}")
            
            # Process the data manually to avoid timezone issues
            from collections import defaultdict
            weekly_symptom_counts = defaultdict(lambda: defaultdict(int))
            unique_symptoms = set()
            
            for log in symptom_logs_raw:
                # Extract date and calculate week start
                log_date = log['date_logged'].date() if hasattr(log['date_logged'], 'date') else log['date_logged']
                week_start = log_date - timedelta(days=log_date.weekday())
                week_str = week_start.strftime('%Y-%m-%d')
                
                symptom = log['symptom'] or ''
                unique_symptoms.add(symptom)
                weekly_symptom_counts[week_str][symptom] += 1
            
            
            # Generate complete week range from start to end
            all_categories = list(symptom_categories.keys()) + ['Other']
            weekly_data = {}
            
            # Initialize all weeks with zero counts
            current_week = start_date - timedelta(days=start_date.weekday())  # Start of week
            while current_week <= end_date:
                week_str = current_week.strftime('%Y-%m-%d')
                weekly_data[week_str] = {cat: 0 for cat in all_categories}
                current_week += timedelta(weeks=1)
            
            # Fill in actual data from our manually processed counts
            categorization_debug = defaultdict(int)
            total_symptoms_processed = 0
            
            for week_str, symptom_counts in weekly_symptom_counts.items():
                if week_str in weekly_data:
                    for symptom_text, count in symptom_counts.items():
                        symptom_lower = symptom_text.lower() if symptom_text else ''
                        total_symptoms_processed += count
                        
                        # Categorize the symptom
                        category = 'Other'  # Default category
                        for cat_name, keywords in symptom_categories.items():
                            if any(keyword.lower() in symptom_lower for keyword in keywords):
                                category = cat_name
                                break
                        
                        # Add to existing week data
                        weekly_data[week_str][category] += count
            
            # Format data for Nivo line chart
            chart_data = []
            all_categories = list(symptom_categories.keys()) + ['Other']
            
            for category in all_categories:
                category_data = []
                for week_str in sorted(weekly_data.keys()):
                    category_data.append({
                        'x': week_str,
                        'y': weekly_data[week_str].get(category, 0)
                    })
                
                chart_data.append({
                    'id': category,
                    'data': category_data
                })
        
            return Response({
                'data': chart_data,
                'date_range': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                'total_weeks': len(weekly_data)
            })
        
        except Exception as e:
            print(f"❌ Error in symptom_trends: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Return empty data structure on error
            return Response({
                'data': [],
                'date_range': {
                    'start': timezone.now().date().isoformat(),
                    'end': timezone.now().date().isoformat()
                },
                'total_weeks': 0,
                'error': str(e)
            })

    @action(detail=False, methods=['get'])
    def symptom_category_counts(self, request):
        """Get total counts for each symptom category for bar chart"""
        try:
            # Define symptom categories mapping
            symptom_categories = {
                'Respiratory': [
                    'cold', 'cough', 'asthma', 'runny nose', 'sore throat', 'itchy throat', 
                    'auri', 'shortness of breath', 'hyperventilation', 'earache', 'nosebleed',
                    'hypertension'
                ],
                'Digestive': [
                    'stomach ache', 'hyperacidity', 'lbm', 'diarrhea', 'vomiting', 
                    'epigastric pain', 'dry mouth'
                ],
                'Pain & Musculoskeletal': [
                    'headache', 'body pain', 'muscle strain', 'chest pain', 'toothache', 
                    'dysmenorrhea', 'cramps', 'menstrual cramps', 'stiff neck', 'migraine',
                    'sprain'
                ],
                'Dermatological & Trauma': [
                    'abrasion', 'wound', 'punctured wound', 'cut', 'pimple', 
                    'hematoma', 'stitches'
                ],
                'Neurological & Psychological': [
                    'dizziness', 'anxiety'
                ],
                'Systemic & Infectious': [
                    'fever', 'malaise', 'infection', 'uti', 'clammy skin', 'allergy', 
                    'skin allergy'
                ]
            }
            
            # Get all symptom logs
            all_symptoms = SymptomLog.objects.values('symptom')
            
            # Count symptoms by category
            category_counts = {}
            for category in symptom_categories.keys():
                category_counts[category] = 0
            category_counts['Other'] = 0
            
            # Process each symptom
            for symptom_log in all_symptoms:
                symptom_text = (symptom_log['symptom'] or '').lower()
                
                # Categorize the symptom
                categorized = False
                for category, keywords in symptom_categories.items():
                    if any(keyword.lower() in symptom_text for keyword in keywords):
                        category_counts[category] += 1
                        categorized = True
                        break
                
                if not categorized:
                    category_counts['Other'] += 1
            
            # Format for Nivo bar chart
            # Create data structure: [{ category: "Respiratory", count: 150 }, ...]
            chart_data = []
            for category, count in category_counts.items():
                if count > 0:  # Only include categories with data
                    chart_data.append({
                        'category': category,
                        'count': count
                    })
            
            # Sort by count (highest first)
            chart_data.sort(key=lambda x: x['count'], reverse=True)
            
            return Response({
                'data': chart_data,
                'total_symptoms': sum(category_counts.values()),
                'categories': list(category_counts.keys())
            })
            
        except Exception as e:
            print(f"❌ Error in symptom_category_counts: {str(e)}")
            return Response({
                'data': [],
                'total_symptoms': 0,
                'categories': [],
                'error': str(e)
            })