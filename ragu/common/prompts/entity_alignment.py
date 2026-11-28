from pydantic import BaseModel, Field

from ragu.common.prompts.messages import ChatMessages, UserMessage
from ragu.common.prompts.prompt_storage import RAGUInstruction


class EntityDecision(BaseModel):
    candidate_id: str = Field(..., description="ID of the candidate entity")
    same_entity: bool = Field(..., description="Whether the candidate and source are the same real-world entity")


class EntityDecisionBatch(BaseModel):
    decisions: list[EntityDecision] = Field(default_factory=list)


ENTITY_ALIGNMENT_INSTRUCTION = RAGUInstruction(
    messages=ChatMessages.from_messages([
        UserMessage(content="""
Decide whether each candidate denotes the same real-world entity as the source entity.
Names may be aliases, abbreviations, transliterations, or spelling variants.
Similar or related entities are not necessarily identical. Use the descriptions as evidence.
Return one decision for every candidate and copy candidate IDs exactly.

Source entity:
ID: {{ entity_id }}
Name: {{ entity_name }}
Type: {{ entity_type }}
Description: {{ entity_description }}

Candidates:
{{ candidates_json }}

Return only valid JSON matching the provided schema.
"""),
    ]),
    pydantic_model=EntityDecisionBatch,
    description="Prompt for deciding whether entity candidates are aliases.",
)
