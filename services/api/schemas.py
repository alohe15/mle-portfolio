"""Pydantic request/response models for the fraud inference API.

In production, historical features (D*, V*, C*) would be populated by a
feature store lookup keyed on card/device identity. In this demo API, the
caller supplies them directly.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TransactionMetadata(BaseModel):
    """Optional caller context — never echoed back in the response."""

    TransactionID: Optional[int] = None
    notes: Optional[str] = None


class TransactionRequest(BaseModel):
    """Raw transaction payload for lgbm_v9 / dataset_v5.

    In production, historical features (D*, V*, C*) would be populated by a
    feature store lookup keyed on card/device identity. In this demo API, the
    caller supplies them directly.

    Current-transaction fields (required): available from the payment event
    itself — amount, product, card/address/email/distance, and TransactionDT
    (needed to compute Dn features).

    Historical / device features (optional extras): D1–D15, C*, V*, M*, id_*,
    DeviceType, DeviceInfo. Omitted values are treated as null and NaN-filled
    after the transformation pipeline.
    """

    model_config = ConfigDict(extra="allow")

    # --- Current-transaction fields ---
    TransactionAmt: float = Field(..., description="Transaction amount (must be > 0)")
    ProductCD: Optional[str] = None
    card1: Optional[float] = None
    card2: Optional[float] = None
    card3: Optional[float] = None
    card4: Optional[str] = None
    card5: Optional[float] = None
    card6: Optional[str] = None
    addr1: Optional[float] = None
    addr2: Optional[float] = None
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    dist1: Optional[float] = None
    dist2: Optional[float] = None
    TransactionDT: Optional[float] = Field(
        None,
        description="Seconds from reference epoch; required to compute Dn features",
    )

    metadata: Optional[TransactionMetadata] = None

    @field_validator("TransactionAmt")
    @classmethod
    def amount_must_be_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("TransactionAmt must be > 0")
        return value

    def feature_dict(self) -> dict[str, Any]:
        """Flat feature map for the predictor (excludes metadata)."""
        data = self.model_dump(exclude={"metadata"}, exclude_none=False)
        # Drop keys that are purely nested metadata remnants.
        return data


class TransactionResponse(BaseModel):
    """Privacy-safe prediction response — no raw inputs or PII."""

    fraud_probability: float
    recommended_action: str
    action_costs: dict[str, float]
    model_version: str
    dataset_version: int
    request_id: str
