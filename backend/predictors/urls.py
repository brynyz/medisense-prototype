from django.urls import path
from . import views

urlpatterns = [
    path('', views.predictors_dashboard, name='predictors_dashboard'),
    path('api/token/', views.get_streamlit_token, name='streamlit_token'),
]
