"""FastAPI authentication dependencies."""

from __future__ import annotations

from typing import Optional

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader, APIKeyQuery, HTTPAuthorizationCredentials, HTTPBearer

from .config import get_auth_config
from .database import auth_db
from .models import UserRecord, UserRole
from .utils import decode_token, verify_api_key

# Security schemes
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)
http_bearer = HTTPBearer(auto_error=False)


async def get_current_user_from_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(http_bearer),
) -> UserRecord:
    """Get current user from JWT access token."""
    config = get_auth_config()
    if not config.enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is disabled",
        )

    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    payload = decode_token(token)

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = await auth_db.get_user_by_id(user_id)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


async def get_current_user_from_api_key(
    request: Request,
    api_key_header: Optional[str] = Security(api_key_header),
    api_key_query: Optional[str] = Security(api_key_query),
) -> UserRecord:
    """Get current user from API key (header or query param)."""
    config = get_auth_config()
    if not config.enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is disabled",
        )

    # Try header first, then query param
    api_key = api_key_header or api_key_query

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "API-Key"},
        )

    # Look up API key
    from .utils import hash_token
    key_hash = hash_token(api_key)

    api_key_record = await auth_db.get_api_key_by_hash(key_hash)
    if not api_key_record or not api_key_record.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    # Check expiration
    if api_key_record.expires_at and api_key_record.expires_at < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key expired",
        )

    # Get user
    user = await auth_db.get_user_by_id(api_key_record.user_id)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )

    # Update last used timestamp
    # (fire and forget - don't await)
    from .database import auth_db as db
    import asyncio
    asyncio.create_task(db.update_api_key_last_used(api_key_record.id))

    return user


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(http_bearer),
    api_key_header: Optional[str] = Security(api_key_header),
    api_key_query: Optional[str] = Security(api_key_query),
) -> UserRecord:
    """Get current user from either JWT token or API key."""
    config = get_auth_config()
    if not config.enabled:
        # If auth is disabled, return a default service user
        # In production, you might want to handle this differently
        return UserRecord(
            id="00000000-0000-0000-0000-000000000000",
            username="anonymous",
            email="anonymous@geomind.local",
            role=UserRole.VIEWER,
            is_active=True,
            created_at=datetime.utcnow(),
        )

    # Try JWT token first
    if credentials:
        try:
            return await get_current_user_from_token(credentials)
        except HTTPException:
            pass  # Fall through to API key

    # Try API key
    if api_key_header or api_key_query:
        try:
            return await get_current_user_from_api_key(
                Request(scope=request.scope, receive=request._receive),
                api_key_header,
                api_key_query,
            )
        except HTTPException:
            pass

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_role(*allowed_roles: UserRole):
    """Dependency that requires specific role(s)."""
    async def role_checker(user: UserRecord = Depends(get_current_user)) -> UserRecord:
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required roles: {[r.value for r in allowed_roles]}",
            )
        return user
    return role_checker


# Convenience role dependencies
require_admin = require_role(UserRole.ADMIN)
require_user_or_admin = require_role(UserRole.USER, UserRole.ADMIN)
require_service = require_role(UserRole.SERVICE)