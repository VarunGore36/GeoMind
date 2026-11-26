"""Authentication module for GeoMind."""

from .config import AuthConfig, get_auth_config
from .database import AuthDatabase, auth_db
from .models import (
    UserRole,
    TokenType,
    User,
    UserCreate,
    UserUpdate,
    UserResponse,
    TokenRequest,
    TokenResponse,
    RefreshTokenRequest,
    APIKeyCreate,
    APIKeyResponse,
)
from .utils import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
    create_api_key,
    verify_api_key,
)
from .dependencies import (
    get_current_user,
    get_current_user_from_token,
    get_current_user_from_api_key,
    require_role,
    require_admin,
    require_user_or_admin,
    require_service,
)
from .api import router as auth_router

__version__ = "0.1.0"

__all__ = [
    # Config
    "AuthConfig",
    "get_auth_config",
    # Database
    "AuthDatabase",
    "auth_db",
    # Models
    "UserRole",
    "TokenType",
    "User",
    "UserCreate",
    "UserUpdate",
    "UserResponse",
    "TokenRequest",
    "TokenResponse",
    "RefreshTokenRequest",
    "APIKeyCreate",
    "APIKeyResponse",
    # Utils
    "hash_password",
    "verify_password",
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "create_api_key",
    "verify_api_key",
    # Dependencies
    "get_current_user",
    "get_current_user_from_token",
    "get_current_user_from_api_key",
    "require_role",
    "require_admin",
    "require_user_or_admin",
    "require_service",
    # API
    "auth_router",
]