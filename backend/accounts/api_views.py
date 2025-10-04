from rest_framework import generics, status, viewsets
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from django.http import JsonResponse
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import get_user_model
from .serializers import (
    UserRegistrationSerializer, 
    UserSerializer, 
    LoginSerializer, 
    ActivityLogSerializer
)
from .models import ActivityLog
from django.db.models import Count
from django.utils import timezone
from datetime import timedelta
import os
from django.core.exceptions import ValidationError

User = get_user_model()


class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    permission_classes = (AllowAny,)
    serializer_class = UserRegistrationSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        
        # Create JWT tokens
        refresh = RefreshToken.for_user(user)
        
        return Response({
            'user': UserSerializer(user).data,
            'refresh': str(refresh),
            'access': str(refresh.access_token),
        }, status=status.HTTP_201_CREATED)


class LoginView(generics.GenericAPIView):
    permission_classes = (AllowAny,)
    serializer_class = LoginSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data['user']
        
        # Create JWT tokens
        refresh = RefreshToken.for_user(user)
        
        # Log activity
        ActivityLog.objects.create(
            user=user,
            action='Login',
            description=f'User {user.username} logged in via API'
        )
        
        return Response({
            'user': UserSerializer(user).data,
            'refresh': str(refresh),
            'access': str(refresh.access_token),
        })


class UserProfileView(generics.RetrieveUpdateAPIView):
    serializer_class = UserSerializer
    permission_classes = [IsAuthenticated]

    def get_object(self):
        return self.request.user


class ActivityLogViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = ActivityLogSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return ActivityLog.objects.filter(user=self.request.user).order_by('-timestamp')


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def logout_view(request):
    try:
        refresh_token = request.data["refresh"]
        token = RefreshToken(refresh_token)
        token.blacklist()
        
        # Log activity
        ActivityLog.objects.create(
            user=request.user,
            action='Logout',
            description=f'User {request.user.username} logged out via API'
        )
        
        return Response({"message": "Successfully logged out"}, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def dashboard_stats(request):
    """Get dashboard statistics for the current user"""
    # This would integrate with your existing analytics
    stats = {
        'patient_trends': [65, 59, 80, 81, 56, 55, 40],
        'inventory_data': [28, 48, 40, 19, 86, 27, 90],
        'total_patients': 1200,
        'recent_activities': ActivityLogSerializer(
            ActivityLog.objects.filter(user=request.user).order_by('-timestamp')[:5],
            many=True
        ).data
    }
    return Response(stats)

# In your Django views
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_user(request):
    try:
        user = request.user
        print(f"Debug: User found: {user.username}")
        print(f"Debug: User type: {type(user)}")
        print(f"Debug: Has profile_image: {hasattr(user, 'profile_image')}")
        
        # Build the response data safely
        response_data = {
            'username': user.username,
            'first_name': user.first_name or '',
            'last_name': user.last_name or '',
            'email': user.email or '',
            'is_staff': user.is_staff,
            'is_active': user.is_active,
            'date_joined': user.date_joined,
            'last_login': user.last_login,
            'role': getattr(user, 'role', 'User'),
            'phone_number': getattr(user, 'phone_number', None),
            'bio': getattr(user, 'bio', None),
        }
        
        # Handle profile image safely
        try:
            if hasattr(user, 'profile_image') and user.profile_image:
                response_data['profile_image'] = request.build_absolute_uri(user.profile_image.url)
            else:
                response_data['profile_image'] = None
        except Exception as img_error:
            print(f"Debug: Profile image error: {img_error}")
            response_data['profile_image'] = None
        
        print(f"Debug: Response data: {response_data}")
        return Response(response_data)
        
    except Exception as e:
        print(f"Debug: Error in get_user: {str(e)}")
        return Response({
            'error': f'Failed to get user data: {str(e)}'
        }, status=500)

@api_view(['PUT'])
@permission_classes([IsAuthenticated])
def update_user_profile(request):
    try:
        user = request.user
        print(f"Debug: Updating profile for user: {user.username}")
        
        # Update basic fields
        user.first_name = request.data.get('first_name', user.first_name)
        user.last_name = request.data.get('last_name', user.last_name)
        user.email = request.data.get('email', user.email)
        user.username = request.data.get('username', user.username)
        
        # Update optional fields
        if 'phone_number' in request.data:
            user.phone_number = request.data.get('phone_number')
        if 'bio' in request.data:
            user.bio = request.data.get('bio')
        
        # Handle profile image upload if provided
        image_info = None
        if 'profile_image' in request.FILES:
            from .utils import validate_image_file, get_image_info, PROFILE_IMAGE_SETTINGS
            
            uploaded_image = request.FILES['profile_image']
            print(f"Debug: Processing image: {uploaded_image.name} ({uploaded_image.size} bytes)")
            
            # Validate the image first
            validation_result = validate_image_file(
                uploaded_image,
                max_size_mb=PROFILE_IMAGE_SETTINGS['max_file_size_mb'],
                allowed_formats=PROFILE_IMAGE_SETTINGS['allowed_formats']
            )
            
            if not validation_result['valid']:
                return Response({
                    'error': f'Image validation failed: {validation_result["error"]}'
                }, status=400)
            
            # Get original image info
            original_info = validation_result['info']
            print(f"Debug: Original image - {original_info['format']} {original_info['size']} {original_info['mode']}")
            
            # Delete old image if exists
            if user.profile_image:
                try:
                    old_path = user.profile_image.path
                    if os.path.exists(old_path):
                        os.remove(old_path)
                        print(f"Debug: Deleted old image: {old_path}")
                except Exception as e:
                    print(f"Debug: Could not delete old image: {e}")
            
            # Set the new image (model will process it automatically)
            user.profile_image = uploaded_image
            
            # Get processed image info after save
            image_info = {
                'original': original_info,
                'processed': 'Will be processed on save'
            }
        
        # Save the user (this will trigger image processing)
        user.save()
        print(f"Debug: User saved successfully")
        
        # Get final image info if image was processed
        if image_info and user.profile_image:
            try:
                processed_info = user.get_profile_image_info()
                if processed_info:
                    image_info['processed'] = processed_info
                    print(f"Debug: Processed image - {processed_info}")
            except Exception as e:
                print(f"Debug: Could not get processed image info: {e}")
        
        # Build response
        response_data = {
            'username': user.username,
            'first_name': user.first_name,
            'last_name': user.last_name,
            'email': user.email,
            'is_staff': user.is_staff,
            'is_active': user.is_active,
            'date_joined': user.date_joined,
            'last_login': user.last_login,
            'profile_image': request.build_absolute_uri(user.profile_image.url) if user.profile_image else None,
            'phone_number': getattr(user, 'phone_number', None),
            'bio': getattr(user, 'bio', None),
        }
        
        # Add image processing info if available
        if image_info:
            response_data['image_processing'] = image_info
        
        print(f"Debug: Response data prepared")
        return Response(response_data)
        
    except ValidationError as e:
        print(f"Debug: Validation error: {e}")
        return Response({
            'error': f'Validation error: {str(e)}'
        }, status=400)
    except Exception as e:
        print(f"Debug: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return Response({
            'error': f'Failed to update profile: {str(e)}'
        }, status=500)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def logout_user(request):
    # Handle logout logic (blacklist token, etc.)
    return Response({'message': 'Logged out successfully'})

@api_view(['GET'])
@permission_classes([AllowAny])
def health_check(request):
    """Health check endpoint for DigitalOcean deployment monitoring"""
    try:
        # Check database connectivity
        User.objects.count()
        
        return JsonResponse({
            'status': 'healthy',
            'timestamp': timezone.now().isoformat(),
            'version': '1.0.0',
            'database': 'connected'
        })
    except Exception as e:
        return JsonResponse({
            'status': 'unhealthy',
            'timestamp': timezone.now().isoformat(),
            'error': str(e)
        }, status=503)
