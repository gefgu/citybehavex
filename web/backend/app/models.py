"""Shared response envelope.

Mirrors the reference project: every endpoint returns ``{"data": ...}`` so the
frontend always reads ``body.data``.
"""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ApiResponseWrapper(BaseModel, Generic[T]):
    data: T = Field(..., description="Response payload")
