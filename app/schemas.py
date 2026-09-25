from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    make: str = Field(min_length=1, max_length=80)
    model: str = Field(min_length=1, max_length=100)
    variant: str = Field(default="", max_length=120)
    year: int = Field(ge=1980, le=2100)
    mileage: float = Field(ge=0, le=1_000_000)
    engineCapacity: float = Field(ge=300, le=10_000)
    transmission: str = Field(default="", max_length=40)
    fuelType: str = Field(default="", max_length=40)
    city: str = Field(default="", max_length=100)
    registrationCity: str = Field(default="", max_length=100)
    condition: str = Field(default="Unknown", max_length=50)
    assemblyType: str = Field(default="Unknown", max_length=50)
    bodyType: str = Field(default="", max_length=60)
    colour: str = Field(default="", max_length=50)
    inspectionScore: float | None = Field(default=None, ge=0, le=100)
    numberOfOwners: float | None = Field(default=None, ge=0, le=20)

    @field_validator("make", "model", "variant", "transmission", "fuelType", "city", "registrationCity", "condition", "assemblyType", "bodyType", "colour")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return " ".join(value.strip().split()).title()


class TrainingRequest(BaseModel):
    records: list[dict[str, Any]] = Field(min_length=1, max_length=100_000)
    provenance: dict[str, Any] | None = None
