"""Typed boundary for sessions accepted by the generator's final model judge."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class GeneratedMessage(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class GateAcceptedSession(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)
    scenario: str
    messages: tuple[GeneratedMessage, ...] = Field(min_length=2)
