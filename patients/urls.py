from django.urls import path
from . import views

urlpatterns = [
    path('', views.symptom_logs_table, name='symptom-logs'),  # This matches the URL name you're using
    path('add/', views.add_symptom_log, name='add_symptom_log'),
    path('edit/<int:log_id>/', views.edit_symptom_log, name='edit_symptom_log'),
    path('delete/<int:log_id>/', views.delete_symptom_log, name='delete_symptom_log'),
    path('export/excel/', views.export_symptom_logs_excel, name='export_symptom_logs_excel'),
]
