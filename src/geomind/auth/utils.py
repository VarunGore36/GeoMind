"""Authentication utilities: JWT, password hashing, API key generation."""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import jwt
from passlib.context import CryptContext

from .config import get_auth_config


# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(
    subject: str,
    roles: list[str],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create a JWT access token."""
    config = get_auth_config()
    jwt_config = config.jwt

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=jwt_config.access_token_expire_minutes)

    to_encode = {
        "sub": subject,
        "roles": roles,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "iss": jwt_config.issuer,
        "aud": jwt_config.audience,
        "type": "access",
    }

    encoded_jwt = jwt.encode(to_encode, jwt_config.secret_key, algorithm=jwt_config.algorithm)
    return encoded_jwt


def create_refresh_token(
    subject: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create a JWT refresh token."""
    config = get_auth_config()
    jwt_config = config.jwt

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(days=jwt_config.refresh_token_expire_days)

    to_encode = {
        "sub": subject,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "iss": jwt_config.issuer,
        "aud": jwt_config.audience,
        "type": "refresh",
    }

    encoded_jwt = jwt.encode(to_encode, jwt_config.secret_key, algorithm=jwt_config.algorithm)
    return encoded_jwt


def decode_token(token: str) -> Optional[dict]:
    """Decode and validate a JWT token."""
    config = get_auth_config()
    jwt_config = config.jwt

    try:
        payload = jwt.decode(
            token,
            jwt_config.secret_key,
            algorithms=[jwt_config.algorithm],
            audience=jwt_config.audience,
            issuer=jwt_config.issuer,
        )
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def create_api_key(prefix: str = "gm_") -> tuple[str, str, str]:
    """
    Generate a new API key.

    Returns:
        tuple: (full_key, key_hash, key_prefix)
    """
    config = get_auth_config()
    api_key_config = config.api_key

    # Generate random key
    raw_key = secrets.token_urlsafe(api_key_config.key_length)
    full_key = f"{prefix}{raw_key}"

    # Hash for storage
    key_hash = hashlib.sha256(full_key.encode()).hexdigest()
    key_prefix = full_key[:api_key_config.key_length] if len(full_key) > len(prefix) else full_key

    return full_key, key_hash, key_prefix


def verify_api_key(provided_key: str, stored_hash: str) -> bool:
    """Verify an API key against its stored hash."""
    provided_hash = hashlib.sha256(provided_key.encode()).hexdigest()
    return secrets.compare_digest(provided_hash, stored_hash)


def hash_token(token: str) -> str:
    """Hash a token for storage."""
    return hashlib.sha256(token.encode()).hexdigest()


def verify_token(token: str, stored_hash: str) -> bool:
    """Verify a token against its stored hash."""
    token_hash = hash_token(token)
    return secrets.compare_digest(token_hash, stored_hash)