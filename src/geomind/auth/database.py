"""Authentication database layer."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator, Optional
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .config import get_auth_config
from .models import Base, RefreshTokenRecord, UserRecord, APIKeyRecord, APIKeyCreate, UserCreate


class AuthDatabase:
    """Async database layer for authentication."""

    def __init__(self, url: str | None = None):
        config = get_auth_config()
        self.engine = create_async_engine(url or "sqlite+aiosqlite:///./auth.db", echo=False)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    async def init(self) -> None:
        """Initialize database tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        """Close database connections."""
        await self.engine.dispose()

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session."""
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    # --- User Management ---

    async def create_user(self, user_data: UserCreate, password_hash: str) -> UserRecord:
        """Create a new user."""
        async with self.session() as session:
            user = UserRecord(
                username=user_data.username,
                email=user_data.email,
                password_hash=password_hash,
                role=user_data.role.value,
            )
            session.add(user)
            await session.flush()
            return user

    async def get_user_by_username(self, username: str) -> Optional[UserRecord]:
        """Get user by username."""
        async with self.session() as session:
            result = await session.execute(
                select(UserRecord).where(UserRecord.username == username)
            )
            return result.scalar_one_or_none()

    async def get_user_by_email(self, email: str) -> Optional[UserRecord]:
        """Get user by email."""
        async with self.session() as session:
            result = await session.execute(
                select(UserRecord).where(UserRecord.email == email)
            )
            return result.scalar_one_or_none()

    async def get_user_by_id(self, user_id: UUID) -> Optional[UserRecord]:
        """Get user by ID."""
        async with self.session() as session:
            result = await session.execute(
                select(UserRecord).where(UserRecord.id == user_id)
            )
            return result.scalar_one_or_none()

    async def update_user(
        self,
        user_id: UUID,
        email: Optional[str] = None,
        role: Optional[str] = None,
        is_active: Optional[bool] = None,
    ) -> Optional[UserRecord]:
        """Update user fields."""
        async with self.session() as session:
            result = await session.execute(
                select(UserRecord).where(UserRecord.id == user_id)
            )
            user = result.scalar_one_or_none()
            if not user:
                return None

            if email is not None:
                user.email = email
            if role is not None:
                user.role = role
            if is_active is not None:
                user.is_active = is_active
            user.updated_at = datetime.utcnow()

            await session.flush()
            return user

    async def update_last_login(self, user_id: UUID) -> None:
        """Update user's last login timestamp."""
        async with self.session() as session:
            await session.execute(
                update(UserRecord)
                .where(UserRecord.id == user_id)
                .values(last_login=datetime.utcnow())
            )

    async def delete_user(self, user_id: UUID) -> bool:
        """Delete a user."""
        async with self.session() as session:
            result = await session.execute(
                select(UserRecord).where(UserRecord.id == user_id)
            )
            user = result.scalar_one_or_none()
            if not user:
                return False
            await session.delete(user)
            return True

    # --- API Key Management ---

    async def create_api_key(
        self,
        user_id: UUID,
        key_data: APIKeyCreate,
        key_hash: str,
        key_prefix: str,
    ) -> "APIKeyRecord":
        """Create a new API key."""
        from .models import APIKeyRecord

        async with self.session() as session:
            api_key = APIKeyRecord(
                key_hash=key_hash,
                key_prefix=key_prefix,
                name=key_data.name,
                user_id=user_id,
                expires_at=(
                    datetime.utcnow() + timedelta(days=key_data.expires_in_days)
                    if key_data.expires_in_days
                    else None
                ),
            )
            session.add(api_key)
            await session.flush()
            return api_key

    async def get_api_key_by_hash(self, key_hash: str) -> Optional["APIKeyRecord"]:
        """Get API key by hash."""
        from .models import APIKeyRecord

        async with self.session() as session:
            result = await session.execute(
                select(APIKeyRecord).where(APIKeyRecord.key_hash == key_hash)
            )
            return result.scalar_one_or_none()

    async def get_user_api_keys(self, user_id: UUID) -> list["APIKeyRecord"]:
        """Get all API keys for a user."""
        from .models import APIKeyRecord

        async with self.session() as session:
            result = await session.execute(
                select(APIKeyRecord)
                .where(APIKeyRecord.user_id == user_id)
                .order_by(APIKeyRecord.created_at.desc())
            )
            return list(result.scalars().all())

    async def update_api_key_last_used(self, key_id: UUID) -> None:
        """Update API key last used timestamp."""
        from .models import APIKeyRecord

        async with self.session() as session:
            await session.execute(
                update(APIKeyRecord)
                .where(APIKeyRecord.id == key_id)
                .values(last_used_at=datetime.utcnow())
            )

    async def revoke_api_key(self, key_id: UUID, user_id: UUID) -> bool:
        """Revoke (delete) an API key."""
        from .models import APIKeyRecord

        async with self.session() as session:
            result = await session.execute(
                select(APIKeyRecord).where(
                    APIKeyRecord.id == key_id,
                    APIKeyRecord.user_id == user_id,
                )
            )
            api_key = result.scalar_one_or_none()
            if not api_key:
                return False
            await session.delete(api_key)
            return True

    # --- Refresh Token Management ---

    async def create_refresh_token(
        self,
        user_id: UUID,
        token_hash: str,
        expires_at: datetime,
    ) -> RefreshTokenRecord:
        """Create a new refresh token."""
        async with self.session() as session:
            token = RefreshTokenRecord(
                token_hash=token_hash,
                user_id=user_id,
                expires_at=expires_at,
            )
            session.add(token)
            await session.flush()
            return token

    async def get_refresh_token(self, token_hash: str) -> Optional[RefreshTokenRecord]:
        """Get refresh token by hash."""
        async with self.session() as session:
            result = await session.execute(
                select(RefreshTokenRecord).where(RefreshTokenRecord.token_hash == token_hash)
            )
            return result.scalar_one_or_none()

    async def revoke_refresh_token(self, token_hash: str) -> bool:
        """Revoke a refresh token."""
        async with self.session() as session:
            result = await session.execute(
                select(RefreshTokenRecord).where(RefreshTokenRecord.token_hash == token_hash)
            )
            token = result.scalar_one_or_none()
            if not token:
                return False
            token.revoked_at = datetime.utcnow()
            return True

    async def revoke_all_user_refresh_tokens(self, user_id: UUID) -> int:
        """Revoke all refresh tokens for a user."""
        async with self.session() as session:
            result = await session.execute(
                select(RefreshTokenRecord).where(
                    RefreshTokenRecord.user_id == user_id,
                    RefreshTokenRecord.revoked_at.is_(None),
                )
            )
            tokens = result.scalars().all()
            for token in tokens:
                token.revoked_at = datetime.utcnow()
            return len(tokens)

    async def cleanup_expired_refresh_tokens(self) -> int:
        """Clean up expired refresh tokens."""
        async with self.session() as session:
            result = await session.execute(
                select(RefreshTokenRecord).where(
                    RefreshTokenRecord.expires_at < datetime.utcnow()
                )
            )
            tokens = result.scalars().all()
            for token in tokens:
                await session.delete(token)
            return len(tokens)


auth_db = AuthDatabase()