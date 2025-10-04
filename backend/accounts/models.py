from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils import timezone
from django.conf import settings
from django.core.exceptions import ValidationError
import os

def user_profile_image_path(instance, filename):
    """Generate file path for user profile images"""
    # Get file extension from processed filename
    ext = filename.split('.')[-1].lower()
    # Create filename: user_id.extension
    filename = f'user_{instance.id}.{ext}'
    # Return the full path
    return os.path.join('profile_images/', filename)

def validate_profile_image(image):
    """Validate profile image before saving"""
    from .utils import validate_image_file, PROFILE_IMAGE_SETTINGS
    
    if image:
        validation_result = validate_image_file(
            image, 
            max_size_mb=PROFILE_IMAGE_SETTINGS['max_file_size_mb'],
            allowed_formats=PROFILE_IMAGE_SETTINGS['allowed_formats']
        )
        
        if not validation_result['valid']:
            raise ValidationError(validation_result['error'])

class User(AbstractUser):
    """Extended User model with profile image"""
    profile_image = models.ImageField(
        upload_to=user_profile_image_path,
        null=True,
        blank=True,
        help_text="Profile picture for the user (max 5MB, will be resized to 400x400)",
        validators=[validate_profile_image]
    )
    
    # Add any other custom fields you need
    phone_number = models.CharField(max_length=15, blank=True, null=True)
    bio = models.TextField(max_length=500, blank=True, null=True)
    
    class Meta:
        db_table = 'accounts_user'
        verbose_name = 'User'
        verbose_name_plural = 'Users'
    
    def __str__(self):
        return self.username
    
    def save(self, *args, **kwargs):
        """Override save to process profile image"""
        # Check if this is an update and profile_image has changed
        if self.pk:
            try:
                old_user = User.objects.get(pk=self.pk)
                if old_user.profile_image and old_user.profile_image != self.profile_image:
                    # Delete old image file
                    if os.path.isfile(old_user.profile_image.path):
                        os.remove(old_user.profile_image.path)
            except User.DoesNotExist:
                pass
        
        # Process new image if provided
        if self.profile_image and hasattr(self.profile_image, 'file'):
            from .utils import compress_and_resize_image, PROFILE_IMAGE_SETTINGS
            
            # Process the image
            processed_image = compress_and_resize_image(
                self.profile_image,
                max_size=PROFILE_IMAGE_SETTINGS['max_size'],
                quality=PROFILE_IMAGE_SETTINGS['quality'],
                format=PROFILE_IMAGE_SETTINGS['format']
            )
            
            # Replace the original with processed image
            self.profile_image = processed_image
        
        super().save(*args, **kwargs)
    
    @property
    def profile_image_url(self):
        """Return profile image URL or None"""
        if self.profile_image and hasattr(self.profile_image, 'url'):
            return self.profile_image.url
        return None
    
    def get_profile_image_info(self):
        """Get information about the profile image"""
        if self.profile_image:
            try:
                from .utils import get_image_info
                return get_image_info(self.profile_image)
            except:
                return None
        return None

class ActivityLog(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    action = models.CharField(max_length=255)
    timestamp = models.DateTimeField(default=timezone.now)
    description = models.TextField(blank=True, null=True)