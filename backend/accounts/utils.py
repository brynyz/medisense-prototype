import os
from PIL import Image, ImageOps
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
import io

def compress_and_resize_image(image_file, max_size=(400, 400), quality=85, format='JPEG'):
    """
    Compress and resize an uploaded image file
    
    Args:
        image_file: Django UploadedFile object
        max_size: Tuple of (width, height) for maximum dimensions
        quality: JPEG quality (1-100, higher is better quality)
        format: Output format ('JPEG', 'PNG', 'WEBP')
    
    Returns:
        ContentFile: Processed image as Django ContentFile
    """
    try:
        # Open the image
        with Image.open(image_file) as img:
            # Convert to RGB if necessary (handles RGBA, P mode, etc.)
            if img.mode in ('RGBA', 'LA', 'P'):
                # Create white background for transparency
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Auto-orient based on EXIF data
            img = ImageOps.exif_transpose(img)
            
            # Calculate new size maintaining aspect ratio
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Create output buffer
            output_buffer = io.BytesIO()
            
            # Save with compression
            save_kwargs = {'format': format, 'optimize': True}
            
            if format == 'JPEG':
                save_kwargs.update({
                    'quality': quality,
                    'progressive': True,  # Progressive JPEG for better loading
                })
            elif format == 'PNG':
                save_kwargs.update({
                    'compress_level': 6,  # PNG compression level (0-9)
                })
            elif format == 'WEBP':
                save_kwargs.update({
                    'quality': quality,
                    'method': 6,  # WebP compression method (0-6)
                })
            
            img.save(output_buffer, **save_kwargs)
            output_buffer.seek(0)
            
            # Create ContentFile
            file_extension = format.lower()
            if file_extension == 'jpeg':
                file_extension = 'jpg'
            
            return ContentFile(
                output_buffer.getvalue(),
                name=f"compressed.{file_extension}"
            )
            
    except Exception as e:
        print(f"Error processing image: {e}")
        # Return original file if processing fails
        return image_file

def get_image_info(image_file):
    """Get information about an image file"""
    try:
        with Image.open(image_file) as img:
            return {
                'format': img.format,
                'mode': img.mode,
                'size': img.size,
                'width': img.width,
                'height': img.height,
            }
    except Exception as e:
        return {'error': str(e)}

def validate_image_file(image_file, max_size_mb=5, allowed_formats=None):
    """
    Validate an uploaded image file
    
    Args:
        image_file: Django UploadedFile object
        max_size_mb: Maximum file size in MB
        allowed_formats: List of allowed formats ['JPEG', 'PNG', 'WEBP']
    
    Returns:
        dict: {'valid': bool, 'error': str or None, 'info': dict}
    """
    if allowed_formats is None:
        allowed_formats = ['JPEG', 'PNG', 'WEBP', 'GIF']
    
    # Check file size
    max_size_bytes = max_size_mb * 1024 * 1024
    if image_file.size > max_size_bytes:
        return {
            'valid': False,
            'error': f'File size ({image_file.size / 1024 / 1024:.1f}MB) exceeds maximum allowed size ({max_size_mb}MB)',
            'info': None
        }
    
    # Check if it's a valid image
    try:
        with Image.open(image_file) as img:
            info = {
                'format': img.format,
                'mode': img.mode,
                'size': img.size,
                'width': img.width,
                'height': img.height,
            }
            
            # Check format
            if img.format not in allowed_formats:
                return {
                    'valid': False,
                    'error': f'Format {img.format} not allowed. Allowed formats: {", ".join(allowed_formats)}',
                    'info': info
                }
            
            # Check dimensions (optional)
            if img.width < 50 or img.height < 50:
                return {
                    'valid': False,
                    'error': 'Image too small. Minimum size is 50x50 pixels',
                    'info': info
                }
            
            if img.width > 5000 or img.height > 5000:
                return {
                    'valid': False,
                    'error': 'Image too large. Maximum size is 5000x5000 pixels',
                    'info': info
                }
            
            return {
                'valid': True,
                'error': None,
                'info': info
            }
            
    except Exception as e:
        return {
            'valid': False,
            'error': f'Invalid image file: {str(e)}',
            'info': None
        }

# Profile image specific settings
PROFILE_IMAGE_SETTINGS = {
    'max_size': (400, 400),  # Maximum dimensions
    'quality': 85,           # JPEG quality
    'format': 'JPEG',        # Output format
    'max_file_size_mb': 5,   # Maximum upload size
    'allowed_formats': ['JPEG', 'PNG', 'WEBP', 'GIF']
}
