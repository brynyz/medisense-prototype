#!/usr/bin/env python3
"""
Test script to verify Django-Streamlit authentication setup
"""
import os
import sys
import jwt
from pathlib import Path

# Add Django project to path
django_path = Path(__file__).parent.parent
sys.path.append(str(django_path))

def test_django_connection():
    """Test Django settings loading"""
    try:
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'medisense.settings')
        import django
        from django.conf import settings
        django.setup()
        
        print("✅ Django settings loaded successfully")
        print(f"Secret key (first 10 chars): {settings.SECRET_KEY[:10]}...")
        print(f"Debug mode: {settings.DEBUG}")
        return settings.SECRET_KEY
    except Exception as e:
        print(f"❌ Django settings error: {e}")
        return None

def test_jwt_token(secret_key):
    """Test JWT token generation and verification"""
    if not secret_key:
        print("❌ Cannot test JWT without secret key")
        return
    
    try:
        # Test payload (similar to what Django generates)
        test_payload = {
            'user_id': 1,
            'username': 'testuser',
            'email': 'test@example.com',
            'is_superuser': False,
            'exp': int(time.time()) + 3600,
            'iat': int(time.time())
        }
        
        # Generate token
        token = jwt.encode(test_payload, secret_key, algorithm='HS256')
        print(f"✅ JWT token generated: {token[:50]}...")
        
        # Verify token
        decoded = jwt.decode(token, secret_key, algorithms=['HS256'])
        print(f"✅ JWT token verified: {decoded['username']}")
        
    except Exception as e:
        print(f"❌ JWT error: {e}")

if __name__ == "__main__":
    import time
    
    print("🔍 Testing Django-Streamlit Authentication Setup")
    print("=" * 50)
    
    secret_key = test_django_connection()
    test_jwt_token(secret_key)
    
    print("\n💡 If tests pass, the authentication should work.")
    print("💡 If tests fail, check your Django settings and virtual environment.")
