from django.urls import path
from . import views
from .views import CustomLoginView

urlpatterns = [
    path('', CustomLoginView.as_view(), name='login'),
    path('register/', views.register, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('home', views.home, name='home'),
    path('settings/', views.settings_view, name='settings'),
    path('backup/', views.backup_database, name='backup_database'),
    path('restore/', views.restore_database, name='restore_database'),
    path('profile_settings/', views.profile_settings, name='profile_settings'),
]