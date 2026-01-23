from pydantic import Field, BaseModel, field_validator


class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="The predicted skin lesion class")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the prediction [0.0 - 1.0]")

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")
        return round(v, 4)
