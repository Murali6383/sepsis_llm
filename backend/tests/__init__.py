"""
Backend Tests — SepsisGuardian AI
===================================
Run from backend/ directory:
    pytest tests/ -v --tb=short

These tests use mocked external dependencies (Groq) — no live API calls.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure backend/ is on sys.path
_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
