from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views
from .api_views import PatientViewSet, SymptomLogViewSet

# API Router
router = DefaultRouter()
router.register(r'patients', PatientViewSet)
router.register(r'symptoms', SymptomLogViewSet)

urlpatterns = [
    # API routes (no 'api/' prefix since main urls.py already has 'api/patients/')
    path('', include(router.urls)),
    
    # Existing template-based routes  
    path('', views.symptom_logs_table, name='symptom-logs'),  
    path('add/', views.add_symptom_log, name='add_symptom_log'),
    path('edit/<int:log_id>/', views.edit_symptom_log, name='edit_symptom_log'),
    path('delete/<int:log_id>/', views.delete_symptom_log, name='delete_symptom_log'),
    path('export/excel/', views.export_symptom_logs_excel, name='export_symptom_logs_excel'),
]
