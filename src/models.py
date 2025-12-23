from __future__ import annotations

from typing import List, Optional, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["agent", "customer", "bot", "other"]
    text: str
    timestamp: Optional[str] = None
    is_filler: bool = False


class Conversation(BaseModel):
    conversation_id: str
    channel: str = "call"
    messages: List[Message]
    messages_for_llm: List[Message] = Field(default_factory=list)
    metadata: dict


class FAQEvidence(BaseModel):
    q_turn: List[int] = Field(default_factory=list)
    a_turn: List[int] = Field(default_factory=list)


class FAQCandidate(BaseModel):
    q_raw: str
    q_clean: str
    a_candidates: List[str] = Field(default_factory=list)
    evidence: FAQEvidence = Field(default_factory=FAQEvidence)


class FlowResolution(BaseModel):
    status: Literal["resolved", "pending", "unresolved", "unknown"] = "unknown"
    summary: str = ""


class FlowEvidence(BaseModel):
    turn_span: List[int] = Field(default_factory=list)


class FlowCandidate(BaseModel):
    intent_candidate: str
    scenario_candidate: str
    flow_steps_candidate: List[str] = Field(default_factory=list)
    resolution: FlowResolution = Field(default_factory=FlowResolution)
    evidence: FlowEvidence = Field(default_factory=FlowEvidence)


class BlueprintSignals(BaseModel):
    policy: List[str] = Field(default_factory=list)
    tone: List[str] = Field(default_factory=list)
    escalations: List[str] = Field(default_factory=list)


class Atoms(BaseModel):
    conversation_id: str
    faq_candidates: List[FAQCandidate] = Field(default_factory=list)
    flow_candidates: List[FlowCandidate] = Field(default_factory=list)
    blueprint_signals: BlueprintSignals = Field(default_factory=BlueprintSignals)


class Ask(BaseModel):
    ask_id: str
    conversation_id: str
    raw_ask: str
    normalized_intent: str
    tags: List[str]


class IntentScenario(BaseModel):
    scenario_name: str
    example_asks: List[str]


class ParentIntent(BaseModel):
    intent_name: str
    definition: str
    scenarios: List[IntentScenario]


class IntentCatalogue(BaseModel):
    parent_intents: List[ParentIntent] = Field(default_factory=list)
    edge_cases: List[dict] = Field(default_factory=list)


class ValueScore(BaseModel):
    zero_to_five: float = Field(alias="0_to_5")
    reason: str


class FAQEntry(BaseModel):
    faq_id: str
    canonical_question: str
    question_variants: List[str]
    draft_answer: str
    needs_verification: bool = True
    value_score: ValueScore


class ToneSummary(BaseModel):
    tone: str
    count: int


class PolicySummary(BaseModel):
    policy: str
    count: int
    examples: List[str] = Field(default_factory=list)


class EscalationPattern(BaseModel):
    pattern: str
    count: int
    examples: List[str] = Field(default_factory=list)


class GapItem(BaseModel):
    faq_id: str
    reason: str


class BlueprintPersona(BaseModel):
    tone_summary: List[ToneSummary] = Field(default_factory=list)
    policy_summary: List[PolicySummary] = Field(default_factory=list)
    escalation_patterns: List[EscalationPattern] = Field(default_factory=list)
    gaps: List[GapItem] = Field(default_factory=list)
