from django.contrib import admin
from .models import InventoryItem

# Register your models here.

@admin.register(InventoryItem)
class InventoryItemAdmin(admin.ModelAdmin):
    list_display = ('name', 'category', 'quantity', 'unit', 'status', 'date_added', 'last_modified_by')
    search_fields = ('name', 'category')
    list_filter = ('category', 'status')
    ordering = ('-date_added',)
    list_per_page = 20

    def last_modified_by(self, obj):
        return obj.last_modified_by.username if obj.last_modified_by else 'N/A'
    
    last_modified_by.short_description = 'Last Modified By'