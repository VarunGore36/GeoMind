"""Authentication API routes."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm

from .config import get_auth_config
from .database import auth_db
from .dependencies import get_current_user, require_admin, require_user_or_admin
from .models import (
    APIKeyCreate,
    APIKeyResponse,
    RefreshTokenRequest,
    TokenRequest,
    TokenResponse,
    UserCreate,
    UserResponse,
    UserRole,
    UserUpdate,
)
from .utils import (
    create_access_token,
    create_api_key,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/login", response_model=TokenResponse, summary="Login with username and password")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Authenticate user and return access + refresh tokens."""
    config = get_auth_config()
    if not config.enabled:
        raise HTTPException(
            status_code=503,
            detail="Authentication is disabled",
        )

    user = await auth_db.get_user_by_username(form_data.username)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    if not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    # Update last login
    await auth_db.update_last_login(user.id)

    # Create tokens
    access_token = create_access_token(
        subject=str(user.id),
        roles=[user.role],
    )
    refresh_token = create_refresh_token(subject=str(user.id))

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=config.jwt.access_token_expire_minutes * 60,
    )


@router.post("/refresh", response_model=TokenResponse, summary="Refresh access token")
async def refresh_token(request: RefreshTokenRequest):
    """Refresh access token using refresh token."""
    config = get_auth_config()
    if not config.enabled:
        raise HTTPException(
            status_code=503,
            detail="Authentication is disabled",
        )

    payload = decode_token(request.refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

    user_id = payload.get("sub")
    user = await auth_db.get_user_by_id(user_id)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )

    # Revoke old refresh token
    from .utils import hash_token
    old_hash = hash_token(request.refresh_token)
    await auth_db.revoke_refresh_token(hash_token(request.refresh_token))

    # Create new tokens
    access_token = create_access_token(subject=str(user.id), roles=[user.role])
    new_refresh_token = create_refresh_token(subject=str(user.id))

    # Save new refresh token
    new_refresh_hash = hash_token(request.refresh_token)  # We'll create new one
    new_expire = datetime.now(timezone.utc) + timedelta(days=7)
    new_refresh_token = create_refresh_token(str(user.id))
    new_refresh_hash = hash_token(new_refresh_token)
    await auth_db.create_refresh_token(user.id, new_refresh_hash, new_expire)

    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=config.jwt.access_token_expire_minutes * 60,
    )


@router.post("/logout", summary="Logout (revoke refresh token)")
async def logout(request: RefreshTokenRequest):
    """Revoke the current refresh token."""
    from .utils import hash_token

    token_hash = hash_token(request.refresh_token)
    revoked = await auth_db.revoke_refresh_token(token_hash)

    return {"revoked": revoked, "message": "Logged out successfully"}


@router.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED, summary="Create new user")
async def create_user(
    user_data: UserCreate,
    current_user = Depends(require_admin),
):
    """Create a new user (admin only)."""
    # Check if user exists
    existing = await auth_db.get_user_by_username(user_data.username)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists",
        )

    existing = await auth_db.get_user_by_email(user_data.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already exists",
        )

    password_hash = hash_password(user_data.password)
    user = await auth_db.create_user(user_data, password_hash)

    return UserResponse(
        id=str(user.id),
        username=user.username,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
        created_at=user.created_at,
        last_login=user.last_login,
    )


@router.get("/users/me", response_model=UserResponse, summary="Get current user profile")
async def get_current_user_profile(current_user = Depends(get_current_user)):
    """Get the current authenticated user's profile."""
    return UserResponse(
        id=str(current_user.id),
        username=current_user.username,
        email=current_user.email,
        role=current_user.role,
        is_active=current_user.is_active,
        created_at=current_user.created_at,
        last_login=current_user.last_login,
    )


@router.patch("/users/me", response_model=UserResponse, summary="Update current user profile")
async def update_current_user_profile(
    user_update: UserUpdate,
    current_user = Depends(get_current_user),
):
    """Update the current user's profile."""
    user = await auth_db.update_user(
        current_user.id,
        email=user_update.email,
        role=user_update.role.value if user_update.role else None,
        is_active=user_update.is_active,
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return UserResponse(
        id=str(user.id),
        username=user.username,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
        created_at=user.created_at,
        last_login=user.last_login,
    )


@router.get("/users", response_model=list[UserResponse], summary="List all users")
async def list_users(
    current_user = Depends(require_admin),
    limit: int = 100,
    offset: int = 0,
):
    """List all users (admin only)."""
    # For now, return empty list - would need a list_users method in auth_db
    return []


@router.post("/api-keys", response_model=APIKeyResponse, status_code=status.HTTP_201_CREATED, summary="Create API key")
async def create_api_key(
    key_data: APIKeyCreate,
    current_user = Depends(get_current_user),
):
    """Create a new API key for the current user."""
    full_key, key_hash, key_prefix = create_api_key()

    api_key_record = await auth_db.create_api_key(
        current_user.id,
        key_data,
        key_hash,
        key_prefix,
    )

    expires_at = None
    if key_data.expires_in_days:
        expires_at = datetime.utcnow() + timedelta(days=key_data.expires_in_days)

    return APIKeyResponse(
        api_key=f"{key_prefix}{secrets.token_urlsafe(32)}",  # Only shown once!
        key_id=str(api_key_record.id),
        name=key_data.name,
        prefix=key_prefix,
        expires_at=expires_at,
        created_at=api_key_record.created_at,
    )


@router.get("/api-keys", summary="List user's API keys")
async def list_api_keys(current_user = Depends(get_current_user)):
    """List all API keys for the current user."""
    keys = await auth_db.get_user_api_keys(current_user.id)
    return [
        {
            "id": str(k.id),
            "name": k.name,
            "prefix": k.key_prefix,
            "is_active": k.is_active,
            "expires_at": k.expires_at,
            "last_used_at": k.last_used_at,
            "created_at": k.created_at,
        }
        for k in keys
    ]


@router.delete("/api-keys/{key_id}", summary="Revoke API key")
async def revoke_api_key(
    key_id: str,
    current_user = Depends(get_current_user),
):
    """Revoke (delete) an API key."""
    from uuid import UUID
    revoked = await auth_db.revoke_api_key(UUID(key_id), current_user.id)
    if not revoked:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )
    return {"message": "API key revoked"}


@router.get("/verify", summary="Verify token")
async def verify_token_endpoint(request: Request):
    """Verify if a token is valid."""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header",
        )

    token = auth_header[7:]  # Remove "Bearer "
    payload = decode_token(token)

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    user_id = payload.get("sub")
    user = await auth_db.get_user_by_id(user_id)

    return {
        "valid": True,
        "user_id": user_id,
        "username": user.username if user else None,
        "roles": payload.get("roles", []),
        "exp": payload.get("exp"),
    }


# Import missing modules at bottom to avoid circular imports
from datetime import datetime
import secrets