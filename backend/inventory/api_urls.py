from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .api_views import InventoryItemViewSet, inventory_stats

# API Router
router = DefaultRouter()
router.register(r'items', InventoryItemViewSet)

urlpatterns = [
    # Router URLs for inventory items
    path('', include(router.urls)),
    
    # Statistics endpoint
    path('stats/', inventory_stats, name='inventory_stats'),
]
