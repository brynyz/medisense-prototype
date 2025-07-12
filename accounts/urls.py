from django.urls import path
from . import views
from .views import CustomLoginView

urlpatterns = [
    path('', CustomLoginView.as_view(), name='login'),
    path('register/', views.register, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('home', views.home, name='home'),
    # path('register/', views.register, name='register'),
]
