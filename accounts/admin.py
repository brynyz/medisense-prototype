from django.contrib import admin
from .models import ActivityLog
from inventory.models import InventoryItem
# Register your models here.


@admin.register(ActivityLog)
class ActivityLogAdmin(admin.ModelAdmin):
    list_display = ('user', 'action', 'timestamp', 'description')
    readonly_fields = ('user', 'action', 'timestamp', 'description')

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False