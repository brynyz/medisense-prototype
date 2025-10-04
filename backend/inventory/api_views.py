from rest_framework import viewsets, permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework import status
from .models import InventoryItem
from .serializers import InventoryItemSerializer

class InventoryItemViewSet(viewsets.ModelViewSet):
    """
    ViewSet for managing inventory items
    """
    queryset = InventoryItem.objects.all()
    serializer_class = InventoryItemSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def perform_create(self, serializer):
        serializer.save(last_modified_by=self.request.user)
    
    def perform_update(self, serializer):
        serializer.save(last_modified_by=self.request.user)

@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def inventory_stats(request):
    """
    Get inventory statistics
    """
    total_items = InventoryItem.objects.count()
    in_stock = InventoryItem.objects.filter(status='In Stock').count()
    low_stock = InventoryItem.objects.filter(status='Low Stock').count()
    out_of_stock = InventoryItem.objects.filter(status='Out of Stock').count()
    
    stats = {
        'total_items': total_items,
        'in_stock': in_stock,
        'low_stock': low_stock,
        'out_of_stock': out_of_stock,
    }
    
    return Response(stats)
