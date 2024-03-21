import jwt
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv('JWT_SECRET_KEY')
ALGORITHM = 'HS256'


def generate_token(user_id):
    """Generate JWT token for authentication.

    Args:
        user_id (str): The unique identifier of the user.

    Returns:
        str: JWT token encoded with user information and expiration time.
    """
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(days=14), 
        'iat': datetime.utcnow(),
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return token

def decode_token(token):
    """Decode and validate JWT token.

    Args:
        token (str): The JWT token to decode.

    Returns:
        dict: Payload information decoded from the token.
        str: Error message if token is expired or invalid.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return 'Token expired. Please log in again.'
    except jwt.InvalidTokenError:
        return 'Invalid token. Please log in again.'


def generate_refresh_token(user_id):
    """Generate JWT refresh token.

    Args:
        user_id (str): The unique identifier of the user.

    Returns:
        str: JWT refresh token encoded with user information and expiration time.
    """
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(days=24), 
        'iat': datetime.utcnow(),
        'type': 'refresh',
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return token

def refresh_access_token(refresh_token):
    """Refresh access token using refresh token.

    Args:
        refresh_token (str): The refresh token used for generating a new access token.

    Returns:
        str: New access token if refresh token is valid.
        str: Error message if refresh token is expired or invalid.
    """
    try:
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get('type') == 'refresh':
            return generate_token(payload['user_id']) 
        else:
            return 'Invalid refresh token.'
    except jwt.ExpiredSignatureError:
        return 'Refresh token expired. Please log in again.'
    except jwt.InvalidTokenError:
        return 'Invalid refresh token. Please log in again.'