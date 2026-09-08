from typing import Literal
from pydantic import BaseModel, Field


class Finding(BaseModel):
    observation: str
    evidence_ids: list[str] = Field(min_length=1)
    severity: Literal["info", "warning", "critical"]


class ColumnMeaning(BaseModel):
    column: str
    description: str
    source: Literal["dictionary", "inferred", "unknown"]


class UnderstandingReport(BaseModel):
    summary: str
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=1)
    proposed_row_grain: str
    glossary: list[ColumnMeaning]
    findings: list[Finding]
    business_questions: list[str]
    limitations: list[str]
