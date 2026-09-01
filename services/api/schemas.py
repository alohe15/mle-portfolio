"""Request and response schemas for the fraud detection API.

The request schema is derived from the feature classification (Step 1).
Raw features are accepted in the request body. Per-row and fitted-lookup
features are computed by the API. Historical features are optional.

§10 compliance: the API accepts raw transaction features, not pre-engineered ones.
"""

from __future__ import annotations

import uuid
from typing import Optional

from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    """Raw transaction input — what a real request looks like.

    Required fields: TransactionAmt (the model cannot score without it).
    All other fields: optional with None defaults → omitted before prediction.
    LightGBM handles NaN natively for missing splits.
    """

    TransactionAmt: float = Field(..., description="Transaction amount in dollars")
    TransactionDT: Optional[float] = Field(
        None,
        description="Time delta in seconds from reference point — required for Dn normalization",
    )

    card1: Optional[float] = None
    card2: Optional[float] = None
    card3: Optional[float] = None
    card4: Optional[str] = None
    card5: Optional[float] = None
    card6: Optional[str] = None

    addr1: Optional[float] = None
    addr2: Optional[float] = None
    dist1: Optional[float] = None
    dist2: Optional[float] = None

    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None

    D1: Optional[float] = None
    D2: Optional[float] = None
    D3: Optional[float] = None
    D4: Optional[float] = None
    D5: Optional[float] = None
    D6: Optional[float] = None
    D7: Optional[float] = None
    D8: Optional[float] = None
    D9: Optional[float] = None
    D10: Optional[float] = None
    D11: Optional[float] = None
    D12: Optional[float] = None
    D13: Optional[float] = None
    D14: Optional[float] = None
    D15: Optional[float] = None

    C1: Optional[float] = None
    C2: Optional[float] = None
    C3: Optional[float] = None
    C4: Optional[float] = None
    C5: Optional[float] = None
    C6: Optional[float] = None
    C7: Optional[float] = None
    C8: Optional[float] = None
    C9: Optional[float] = None
    C10: Optional[float] = None
    C11: Optional[float] = None
    C12: Optional[float] = None
    C13: Optional[float] = None
    C14: Optional[float] = None

    M1: Optional[str] = None
    M2: Optional[str] = None
    M3: Optional[str] = None
    M4: Optional[str] = None
    M5: Optional[str] = None
    M6: Optional[str] = None
    M7: Optional[str] = None
    M8: Optional[str] = None
    M9: Optional[str] = None

    ProductCD: Optional[str] = None

    model_config = {"extra": "allow"}


class PredictionResponse(BaseModel):
    """API response — privacy-safe, no raw input echoed."""

    fraud_probability: float = Field(..., description="Calibrated fraud probability")
    raw_score: float = Field(..., description="Pre-calibration model output")
    recommended_action: str = Field(
        ..., description="approve, step_up, review, or decline"
    )
    model_version: int
    dataset_version: int
    policy_version: int
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    warnings: list[str] = Field(default_factory=list, description="Non-fatal issues")


class HealthResponse(BaseModel):
    status: str
    model_version: int
    dataset_version: int
    policy_version: Optional[int] = None
    n_features: int


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
