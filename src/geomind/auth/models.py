"""Authentication models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, String, Text, Boolean, Index, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()


class UserRole(str, Enum):
    """User roles for RBAC."""

    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"
    SERVICE = "service"  # For machine-to-machine


class TokenType(str, Enum):
    """Token types."""

    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"


class User(BaseModel):
    """User model for API responses."""

    id: str
    username: str
    email: str
    role: UserRole
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None


class UserRecord(Base):
    """User database record."""

    __tablename__ = "users"
    __table_args__ = (
        Index("ix_users_username", "username", unique=True),
        Index("ix_users_email", "email", unique=True),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(64), nullable=False, unique=True, index=True)
    email = Column(String(256), nullable=False, unique=True, index=True)
    password_hash = Column(String(256), nullable=False)
    role = Column(String(16), nullable=False, default=UserRole.USER.value)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    api_keys = relationship("APIKeyRecord", back_populates="user", cascade="all, delete-orphan")
    refresh_tokens = relationship("RefreshTokenRecord", back_populates="user", cascade="all, delete-orphan")


class APIKeyRecord(Base):
    """API Key database record."""

    __tablename__ = "api_keys"
    __table_args__ = (
        Index("ix_api_keys_key_hash", "key_hash", unique=True),
        Index("ix_api_keys_user_id", "user_id"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key_hash = Column(String(64), nullable=False, unique=True, index=True)
    key_prefix = Column(String(16), nullable=False, index=True)  # First 8 chars for identification
    name = Column(String(128), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    expires_at = Column(DateTime, nullable=True)
    last_used_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship
    user = relationship("UserRecord", back_populates="api_keys")


class RefreshTokenRecord(Base):
    """Refresh token database record."""

    __tablename__ = "refresh_tokens"
    __table_args__ = (
        Index("ix_refresh_tokens_token_hash", "token_hash", unique=True),
        Index("ix_refresh_tokens_user_id", "user_id"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    token_hash = Column(String(64), nullable=False, unique=True, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    expires_at = Column(DateTime, nullable=False)
    revoked_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship
    user = relationship("UserRecord", back_populates="refresh_tokens")


# API Request/Response Models
class TokenRequest(BaseModel):
    """Token request payload."""

    username: str
    password: str


class TokenResponse(BaseModel):
    """Token response payload."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshTokenRequest(BaseModel):
    """Refresh token request payload."""

    refresh_token: str


class APIKeyCreate(BaseModel):
    """API Key creation request."""

    name: str = Field(..., min_length=1, max_length=128)
    expires_in_days: Optional[int] = Field(default=None, ge=1)


class APIKeyResponse(BaseModel):
    """API Key creation response (only time the full key is shown)."""

    api_key: str
    key_id: str
    name: str
    prefix: str
    expires_at: Optional[datetime] = None
    created_at: datetime


class UserCreate(BaseModel):
    """User creation request."""

    username: str = Field(..., min_length=3, max_length=64)
    email: str = Field(..., max_length=256)
    password: str = Field(..., min_length=8, max_length=128)
    role: UserRole = UserRole.USER


class UserUpdate(BaseModel):
    """User update request."""

    email: Optional[str] = Field(default=None, max_length=256)
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


class UserResponse(BaseModel):
    """User response model."""

    id: str
    username: str
    email: str
    role: UserRole
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None