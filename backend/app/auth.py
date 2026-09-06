from __future__ import annotations

import base64
import hashlib
import hmac
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jwt
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import InvalidTokenError

from .config import settings
from .schemas import UserPublic


PBKDF2_ITERATIONS = 600_000
bearer_scheme = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    return "pbkdf2_sha256${}${}${}".format(
        PBKDF2_ITERATIONS,
        base64.urlsafe_b64encode(salt).decode("ascii"),
        base64.urlsafe_b64encode(digest).decode("ascii"),
    )


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations, salt_text, expected_text = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        salt = base64.urlsafe_b64decode(salt_text.encode("ascii"))
        expected = base64.urlsafe_b64decode(expected_text.encode("ascii"))
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, int(iterations)
        )
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


@dataclass
class UserStore:
    path: Path

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    username_normalized TEXT NOT NULL UNIQUE,
                    email TEXT NOT NULL,
                    email_normalized TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def create_user(self, username: str, email: str, password: str) -> UserPublic:
        username = username.strip()
        email = email.strip().casefold()
        created_at = datetime.now(timezone.utc).isoformat()
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO users
                        (username, username_normalized, email, email_normalized,
                         password_hash, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        username,
                        username.casefold(),
                        email,
                        email.casefold(),
                        hash_password(password),
                        created_at,
                    ),
                )
                user_id = int(cursor.lastrowid)
        except sqlite3.IntegrityError as error:
            message = str(error)
            detail = "用户名已存在" if "username_normalized" in message else "邮箱已注册"
            raise HTTPException(status_code=409, detail=detail) from error
        return UserPublic(id=user_id, username=username, email=email, created_at=created_at)

    def authenticate(self, account: str, password: str) -> UserPublic | None:
        normalized = account.strip().casefold()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, username, email, password_hash, created_at
                FROM users
                WHERE username_normalized = ? OR email_normalized = ?
                LIMIT 1
                """,
                (normalized, normalized),
            ).fetchone()
        if row is None or not verify_password(password, row["password_hash"]):
            return None
        return self._public(row)

    def get_user(self, user_id: int) -> UserPublic | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, username, email, created_at FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
        return self._public(row) if row else None

    @staticmethod
    def _public(row: sqlite3.Row) -> UserPublic:
        return UserPublic(
            id=int(row["id"]),
            username=str(row["username"]),
            email=str(row["email"]),
            created_at=str(row["created_at"]),
        )


def create_access_token(user: UserPublic) -> tuple[str, int]:
    expires_seconds = settings.jwt_expire_minutes * 60
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user.id),
        "username": user.username,
        "iat": now,
        "exp": now + timedelta(seconds=expires_seconds),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256"), expires_seconds


async def get_current_user(request: Request) -> UserPublic:
    credentials: HTTPAuthorizationCredentials | None = await bearer_scheme(request)
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="请先登录或登录状态已过期",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if credentials is None or credentials.scheme.casefold() != "bearer":
        raise unauthorized
    try:
        payload = jwt.decode(
            credentials.credentials, settings.jwt_secret, algorithms=["HS256"]
        )
        user_id = int(payload["sub"])
    except (InvalidTokenError, KeyError, TypeError, ValueError) as error:
        raise unauthorized from error
    user = request.app.state.user_store.get_user(user_id)
    if user is None:
        raise unauthorized
    return user
