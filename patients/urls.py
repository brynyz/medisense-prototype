from django.urls import path
from . import views

urlpatterns = [
    path('', views.symptom_log_view, name='symptom_log'),
    path('add/', views.add_symptom, name='add_symptom'),
    path('edit/<int:log_id>/', views.edit_symptom, name='edit_symptom'),
    path('delete/<int:log_id>/', views.delete_symptom, name='delete_symptom'),
]
