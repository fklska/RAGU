import json
import math
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
from ragu.models.decision_client import DecisionClient, DecisionRequest
from ragu.models.llm import LLM

__all__ = [
    "EntityDecider",
    "EntityDecision",
    "EntityDecisionBatch",
    "FridaEntityDecider",
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


def _entity_payload(entity: Entity) -> dict[str, str]:
    return {
        "id": entity.id,
        "name": entity.entity_name,
        "type": entity.entity_type,
        "description": entity.description,
    }


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
        candidate_data = [_entity_payload(candidate) for candidate in candidates]
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


class FridaEntityDecider(EntityDecider):
    def __init__(
        self,
        decider_client: DecisionClient,
        threshold: float = 0.5,
    ):
        """
        Decide entity identity with FRIDA-Decisions or a compatible model.

        One request contains the source, all candidates, and one ``noul``
        question per candidate. Candidates with probability greater than or
        equal to ``threshold`` are accepted.

        :param decider_client: Local or HTTP structured decision client.
        :param threshold: Minimum probability of the affirmative answer.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.decider_client = decider_client
        self.threshold = threshold

    async def decide(
        self,
        entity: Entity,
        candidates: list[Entity],
    ) -> list[EntityDecision]:
        if not candidates:
            return []

        question_keys = [f"same_entity_{index}" for index in range(len(candidates))]
        request: DecisionRequest = {
            "state": {
                "source_entity": _entity_payload(entity),
                "candidate_entities": [
                    _entity_payload(candidate) for candidate in candidates
                ],
            },
            "questions": {
                key: {
                    "type": "noul",
                    "instructions": (
                        f"Do source_entity and candidate {index} in "
                        "candidate_entities denote the same real-world entity? "
                        "Names may be aliases, abbreviations, transliterations, "
                        "or spelling variants. Similar or related entities are "
                        "not necessarily identical. Use names, types, and "
                        "descriptions as evidence."
                    ),
                }
                for index, key in enumerate(question_keys)
            },
        }
        response = await self.decider_client.decide(request)
        answers = response.get("answers")
        if not isinstance(answers, dict):
            raise TypeError("Decision model response must contain answers")

        decisions = []
        for key, candidate in zip(question_keys, candidates):
            answer = answers.get(key)
            if not isinstance(answer, dict) or "noul" not in answer:
                raise ValueError(f"Decision model response has no noul answer for {key}")
            try:
                probability = float(answer["noul"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid noul probability for {key}") from error
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"Invalid noul probability for {key}")
            decisions.append(
                EntityDecision(
                    candidate_id=candidate.id,
                    same_entity=probability >= self.threshold,
                )
            )
        return decisions
