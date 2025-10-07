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
    path('api/', include(router.urls)),
    path('register/', RegisterView.as_view(), name='api_register'),
    path('login/', LoginView.as_view(), name='api_login'),
    path('logout/', logout_view, name='api_logout'),
    path('refresh/', TokenRefreshView.as_view(), name='api_token_refresh'),
    path('profile/', UserProfileView.as_view(), name='api_profile'),
    path('dashboard/stats/', dashboard_stats, name='api_dashboard_stats'),
    path('auth/user/', get_user, name='api_get_user'),
    path('auth/logout-user/', logout_user, name='api_logout_user'),
    path('auth/profile/update/', update_user_profile, name='api_update_user_profile'),
    path('auth/health/', health_check, name='api_health_check'),
]