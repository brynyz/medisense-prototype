from django.urls import path, include
from django.contrib.auth import views as auth_views
from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import TokenRefreshView
from . import views
from .api_views import (
    RegisterView, 
    LoginView, 
    UserProfileView, 
    ActivityLogViewSet,
    logout_view,
    dashboard_stats,
    get_user,
    logout_user,
    update_user_profile,
    health_check
)

# API Router
router = DefaultRouter()
router.register(r'activity-logs', ActivityLogViewSet, basename='activitylog')

urlpatterns = [
    
    # API routes
]