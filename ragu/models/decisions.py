import json
from abc import ABC, abstractmethod

from ragu.common.base import RaguGenerativeModule
from ragu.common.prompts.entity_alignment import (
    ENTITY_ALIGNMENT_INSTRUCTION,
    EntityDecision,
    EntityDecisionBatch,
)
from ragu.common.prompts.messages import render
from ragu.common.prompts.prompt_storage import require_prompt_schema
from ragu.graph.types import Entity
from ragu.models.llm import LLM

__all__ = [
    "EntityDecider",
    "EntityDecision",
    "EntityDecisionBatch",
    "LLMEntityAlignmentDecider",
]


class EntityDecider(ABC):
    @abstractmethod
    async def decide(
        self,
        entity: Entity,
        candidates: list[Entity],
    ) -> list[EntityDecision]:
        pass


class LLMEntityAlignmentDecider(RaguGenerativeModule, EntityDecider):
    def __init__(self, llm: LLM):
        RaguGenerativeModule.__init__(
            self,
            prompts={"entity_alignment": ENTITY_ALIGNMENT_INSTRUCTION},
        )
        self.llm = llm

    async def decide(
        self,
        entity: Entity,
        candidates: list[Entity],
    ) -> list[EntityDecision]:
        if not candidates:
            return []

        instruction = self.get_prompt("entity_alignment")
        require_prompt_schema(instruction, "entity_alignment", EntityDecisionBatch)
        candidate_data = [
            {
                "id": candidate.id,
                "name": candidate.entity_name,
                "type": candidate.entity_type,
                "description": candidate.description,
            }
            for candidate in candidates
        ]
        conversation = render(
            instruction.messages,
            entity_id=entity.id,
            entity_name=entity.entity_name,
            entity_type=entity.entity_type,
            entity_description=entity.description,
            candidates_json=json.dumps(candidate_data, ensure_ascii=False, indent=2),
        )[0]
        response = await self.llm.chat_completion(
            conversation=conversation.to_openai(),
            output_schema=EntityDecisionBatch,
        )
        return response.decisions
