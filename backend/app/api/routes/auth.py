"""Authentication routes — demo JWT login for development."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.core.security import create_access_token
from app.schemas.schemas import LoginRequest, TokenResponse

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

# Demo credentials — in production use a real user store.
_DEMO_USERS = {
    "doctor": {"password": "doctor123", "role": "doctor"},
    "nurse": {"password": "nurse123", "role": "nurse"},
    "admin": {"password": "admin123", "role": "admin"},
}


@router.post("/login", response_model=TokenResponse)
def login(request: LoginRequest):
    """
    Authenticate and return a JWT.
    Demo accounts: doctor/doctor123, nurse/nurse123, admin/admin123.
    """
    user = _DEMO_USERS.get(request.username)
    if not user or user["password"] != request.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )
    token = create_access_token(subject=request.username, role=user["role"])
    return TokenResponse(access_token=token, role=user["role"])
