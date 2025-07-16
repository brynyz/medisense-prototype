from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class ActivityLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    action = models.CharField(max_length=255)
    timestamp = models.DateTimeField(default=timezone.now)
    description = models.TextField(blank=True, null=True)