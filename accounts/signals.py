from django.contrib.auth.signals import user_logged_in, user_logged_out
from django.dispatch import receiver
from django.db.models.signals import post_save, post_delete
from .models import ActivityLog
from inventory.models import InventoryItem

@receiver(user_logged_in)
def log_user_login(sender, request, user, **kwargs):
    ActivityLog.objects.create(
        user=user, action="login",
        description=f"User {user.username} logged out"
        )

@receiver(user_logged_out)
def log_user_logout(sender, request, user, **kwargs):
    ActivityLog.objects.create(
        user=user, action="logout",
        description=f"User {user.username} logged out"
        )

@receiver(post_save, sender=InventoryItem)
def log_medicine_save(sender, instance, created, **kwargs):
    action = "added" if created else "updated"
    ActivityLog.objects.create(
        user=instance.last_modified_by,
        action=action,
        description=f"Inventory: {instance.name} ({action})"
    )

@receiver(post_delete, sender=InventoryItem)
def log_medicine_delete(sender, instance, **kwargs):
    ActivityLog.objects.create(
        user=instance.last_modified_by,
        action="deleted",
        description=f"Inventory: {instance.name} (deleted)"
    )
