from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.conf import settings
import jwt
import time


@login_required
def predictors_dashboard(request):
    """
    Main predictors dashboard view that embeds Streamlit app
    """
    # Generate JWT token for Streamlit authentication
    streamlit_token = generate_streamlit_token(request.user)
    
    context = {
        'streamlit_url': f"{settings.STREAMLIT_URL}?token={streamlit_token}",
        'user': request.user,
    }
    return render(request, 'predictors/dashboard.html', context)


def generate_streamlit_token(user):
    """
    Generate JWT token for Streamlit authentication
    """
    payload = {
        'user_id': user.id,
        'username': user.username,
        'email': user.email,
        'is_superuser': user.is_superuser,
        'exp': int(time.time()) + 3600,  # Token expires in 1 hour
        'iat': int(time.time())
    }
    
    # Use Django's SECRET_KEY for JWT signing
    token = jwt.encode(payload, settings.SECRET_KEY, algorithm='HS256')
    return token


@login_required
def get_streamlit_token(request):
    """
    API endpoint to refresh Streamlit token
    """
    token = generate_streamlit_token(request.user)
    return JsonResponse({'token': token})
