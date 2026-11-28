import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from graspologic_native import hierarchical_leiden

from ragu.graph.graph_builder_pipeline import GraphBuilderModule
from ragu.graph.types import Entity, Relation
from ragu.models.decisions import EntityDecider
from ragu.models.embedder import Embedder


@dataclass(slots=True)
class AlignmentReport:
    entities_before: int = 0
    entities_after: int = 0
    candidate_pairs: int = 0
    accepted_pairs: int = 0
    merged_components: int = 0
    rewritten_relations: int = 0
    dropped_self_loops: int = 0
    decision_calls: int = 0


class _UnionFind:
    def __init__(self, values: list[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


class EntityAligner(GraphBuilderModule):
    def __init__(
        self,
        embedder: Embedder,
        decider: EntityDecider,
        candidate_top_k: int = 5,
        max_candidates_per_entity: int = 15,
        ngram_size: int = 3,
        ngram_min_similarity: float = 0.0,
        dense_min_similarity: float = -1.0,
        cluster_scope: Literal["off", "dense"] = "off",
        merge_mode: Literal["append"] = "append",
        random_seed: int = 0,
    ):
        if candidate_top_k < 1:
            raise ValueError("candidate_top_k must be positive")
        if max_candidates_per_entity < 1:
            raise ValueError("max_candidates_per_entity must be positive")
        if ngram_size < 1:
            raise ValueError("ngram_size must be positive")
        if not 0.0 <= ngram_min_similarity <= 1.0:
            raise ValueError("ngram_min_similarity must be between 0 and 1")
        if not -1.0 <= dense_min_similarity <= 1.0:
            raise ValueError("dense_min_similarity must be between -1 and 1")
        if cluster_scope not in ("off", "dense"):
            raise ValueError("cluster_scope must be 'off' or 'dense'")
        if merge_mode != "append":
            raise ValueError("Only merge_mode='append' is supported")

        self.embedder = embedder
        self.decider = decider
        self.candidate_top_k = candidate_top_k
        self.max_candidates_per_entity = max_candidates_per_entity
        self.ngram_size = ngram_size
        self.ngram_min_similarity = ngram_min_similarity
        self.dense_min_similarity = dense_min_similarity
        self.cluster_scope = cluster_scope
        self.merge_mode = merge_mode
        self.random_seed = random_seed
        self.last_report = AlignmentReport()

    async def run(
        self,
        entities: list[Entity],
        relations: list[Relation],
        **kwargs: Any,
    ) -> tuple[list[Entity], list[Relation]]:
        report = AlignmentReport(entities_before=len(entities))
        self.last_report = report
        if len(entities) < 2:
            report.entities_after = len(entities)
            return entities, relations

        entity_ids = [entity.id for entity in entities]
        if len(set(entity_ids)) != len(entity_ids):
            raise ValueError("EntityAligner expects unique entity IDs")

        entity_by_id = {entity.id: entity for entity in entities}
        ngram_candidates = self._ngram_candidates(entities)
        dense_candidates = await self._dense_candidates(entities, relations)
        candidates_by_index: dict[int, list[Entity]] = {}

        for index, entity in enumerate(entities):
            scores: dict[str, float] = {}
            for channel in (ngram_candidates, dense_candidates):
                for candidate_id, score in channel.get(index, {}).items():
                    scores[candidate_id] = max(scores.get(candidate_id, -1.0), score)
            ranked_ids = sorted(
                scores,
                key=lambda candidate_id: (-scores[candidate_id], candidate_id),
            )
            ranked_ids = ranked_ids[:self.max_candidates_per_entity]
            candidates_by_index[index] = [entity_by_id[value] for value in ranked_ids]

        candidate_pairs = {
            self._pair(entity.id, candidate.id)
            for index, entity in enumerate(entities)
            for candidate in candidates_by_index[index]
        }
        report.candidate_pairs = len(candidate_pairs)

        accepted_pairs = set()
        for index, entity in enumerate(entities):
            candidates = candidates_by_index[index]
            if not candidates:
                continue
            report.decision_calls += 1
            decisions = await self.decider.decide(entity, candidates)
            allowed_ids = {candidate.id for candidate in candidates}
            for decision in decisions:
                if not decision.same_entity or decision.candidate_id not in allowed_ids:
                    continue
                if decision.candidate_id == entity.id:
                    continue
                accepted_pairs.add(self._pair(entity.id, decision.candidate_id))

        report.accepted_pairs = len(accepted_pairs)
        if not accepted_pairs:
            report.entities_after = len(entities)
            return entities, relations

        union_find = _UnionFind(entity_ids)
        for left_id, right_id in sorted(accepted_pairs):
            union_find.union(left_id, right_id)

        components: dict[str, list[Entity]] = defaultdict(list)
        for entity in entities:
            components[union_find.find(entity.id)].append(entity)

        canonical_by_old_id: dict[str, str] = {}
        merged_by_canonical_id: dict[str, Entity] = {}
        for component in components.values():
            canonical = self._canonical_entity(component)
            for entity in component:
                canonical_by_old_id[entity.id] = canonical.id
            if len(component) == 1:
                merged_by_canonical_id[canonical.id] = canonical
            else:
                merged_by_canonical_id[canonical.id] = self._merge_entities(component, canonical)

        merged_entities = [
            merged_by_canonical_id[canonical_by_old_id[entity.id]]
            for entity in entities
            if canonical_by_old_id[entity.id] == entity.id
        ]
        merged_relations = self._rewrite_relations(
            relations,
            canonical_by_old_id,
            merged_by_canonical_id,
            report,
        )
        report.entities_after = len(merged_entities)
        report.merged_components = sum(len(component) > 1 for component in components.values())
        return merged_entities, merged_relations

    def _ngram_candidates(self, entities: list[Entity]) -> dict[int, dict[str, float]]:
        normalized_names = [self._normalize(entity.entity_name) for entity in entities]
        entity_ngrams = [self._ngrams(value) for value in normalized_names]
        inverted_index: dict[str, list[int]] = defaultdict(list)
        for index, ngrams in enumerate(entity_ngrams):
            for ngram in ngrams:
                inverted_index[ngram].append(index)

        result: dict[int, dict[str, float]] = {}
        for index, entity in enumerate(entities):
            overlaps: dict[int, int] = defaultdict(int)
            for ngram in entity_ngrams[index]:
                for candidate_index in inverted_index[ngram]:
                    if candidate_index == index:
                        continue
                    if not self._same_type(entity, entities[candidate_index]):
                        continue
                    overlaps[candidate_index] += 1

            scores = {}
            for candidate_index, overlap in overlaps.items():
                denominator = min(
                    len(entity_ngrams[index]),
                    len(entity_ngrams[candidate_index]),
                )
                if denominator == 0:
                    continue
                score = overlap / denominator
                if score > self.ngram_min_similarity:
                    scores[entities[candidate_index].id] = score
            result[index] = self._top_k(scores, entities)
        return result

    async def _dense_candidates(
        self,
        entities: list[Entity],
        relations: list[Relation],
    ) -> dict[int, dict[str, float]]:
        embeddings = await self.embedder.batch_embed_text(
            [entity.to_text() for entity in entities],
            desc="Entity alignment",
        )
        vectors = np.asarray(embeddings, dtype=float)
        if vectors.ndim != 2 or vectors.shape[0] != len(entities):
            raise ValueError("Embedder returned an invalid entity embedding batch")
        if not np.isfinite(vectors).all():
            raise ValueError("Entity embeddings contain non-finite values")

        norms = np.linalg.norm(vectors, axis=1)
        valid = norms > 0
        normalized = np.zeros_like(vectors)
        normalized[valid] = vectors[valid] / norms[valid, None]
        membership = (
            self._community_membership(entities, relations)
            if self.cluster_scope == "dense"
            else {}
        )
        result: dict[int, dict[str, float]] = {}

        for index, entity in enumerate(entities):
            if not valid[index]:
                result[index] = {}
                continue
            similarities = normalized @ normalized[index]
            eligible = []
            for candidate_index, candidate in enumerate(entities):
                if candidate_index == index or not valid[candidate_index]:
                    continue
                if not self._same_type(entity, candidate):
                    continue
                if similarities[candidate_index] < self.dense_min_similarity:
                    continue
                if not self._same_cluster(entity.id, candidate.id, membership):
                    continue
                eligible.append(candidate_index)
            eligible.sort(key=lambda value: (-float(similarities[value]), entities[value].id))
            result[index] = {
                entities[value].id: float(similarities[value])
                for value in eligible[:self.candidate_top_k]
            }
        return result

    def _community_membership(
        self,
        entities: list[Entity],
        relations: list[Relation],
    ) -> dict[str, int]:
        entity_ids = {entity.id for entity in entities}
        edges = {
            self._pair(relation.subject_id, relation.object_id)
            for relation in relations
            if relation.subject_id in entity_ids
            and relation.object_id in entity_ids
            and relation.subject_id != relation.object_id
        }
        if not edges:
            return {}

        weighted_edges = [(left, right, 1.0) for left, right in sorted(edges)]
        raw_mapping = hierarchical_leiden(
            weighted_edges,
            starting_communities=None,
            max_cluster_size=max(2, len(entities)),
            seed=self.random_seed,
            resolution=1.0,
            randomness=0.001,
            use_modularity=True,
            iterations=1,
        )
        membership: dict[str, int] = {}
        levels: dict[str, int] = {}
        for item in raw_mapping:
            node_id, cluster_id, level = self._mapping_item(item)
            if node_id not in entity_ids:
                continue
            if node_id not in levels or level < levels[node_id]:
                membership[node_id] = cluster_id
                levels[node_id] = level
        return membership

    def _rewrite_relations(
        self,
        relations: list[Relation],
        canonical_by_old_id: dict[str, str],
        merged_by_canonical_id: dict[str, Entity],
        report: AlignmentReport,
    ) -> list[Relation]:
        rewritten = []
        for relation in relations:
            subject_id = canonical_by_old_id.get(relation.subject_id, relation.subject_id)
            object_id = canonical_by_old_id.get(relation.object_id, relation.object_id)
            if subject_id == object_id:
                report.dropped_self_loops += 1
                continue

            subject = merged_by_canonical_id.get(subject_id)
            object_ = merged_by_canonical_id.get(object_id)
            subject_name = subject.entity_name if subject else relation.subject_name
            object_name = object_.entity_name if object_ else relation.object_name
            if (
                subject_id == relation.subject_id
                and object_id == relation.object_id
                and subject_name == relation.subject_name
                and object_name == relation.object_name
            ):
                rewritten.append(relation)
                continue

            rewritten.append(
                Relation(
                    id="auto",
                    subject_id=subject_id,
                    object_id=object_id,
                    subject_name=subject_name,
                    object_name=object_name,
                    relation_type=relation.relation_type,
                    description=relation.description,
                    relation_strength=relation.relation_strength,
                    source_chunk_id=list(relation.source_chunk_id),
                )
            )
            report.rewritten_relations += 1
        return rewritten

    @staticmethod
    def _merge_entities(component: list[Entity], canonical: Entity) -> Entity:
        descriptions = []
        seen = set()
        for entity in sorted(component, key=lambda item: (item.entity_name.casefold(), item.id)):
            text = (entity.description or "").strip()
            for part in re.split(r"\n+|(?<=[.!?])\s+", text):
                cleaned = re.sub(r"\s+", " ", part).strip()
                key = cleaned.casefold()
                if cleaned and key not in seen:
                    seen.add(key)
                    descriptions.append(cleaned)

        return Entity(
            id=canonical.id,
            entity_name=canonical.entity_name,
            entity_type=canonical.entity_type,
            description=" ".join(descriptions),
            source_chunk_id=sorted({
                chunk_id
                for entity in component
                for chunk_id in entity.source_chunk_id
            }),
            documents_id=sorted({
                document_id
                for entity in component
                for document_id in entity.documents_id
            }),
            clusters=list(canonical.clusters),
        )

    @staticmethod
    def _canonical_entity(component: list[Entity]) -> Entity:
        return min(
            component,
            key=lambda entity: (
                -len(entity.source_chunk_id),
                -len(entity.entity_name),
                entity.entity_name.casefold(),
                entity.id,
            ),
        )

    @staticmethod
    def _mapping_item(item: Any) -> tuple[str, int, int]:
        if hasattr(item, "node") and hasattr(item, "cluster") and hasattr(item, "level"):
            return str(item.node), int(item.cluster), int(item.level)
        if isinstance(item, dict):
            return str(item["node"]), int(item["cluster"]), int(item["level"])
        if isinstance(item, (tuple, list)) and len(item) >= 3:
            return str(item[0]), int(item[1]), int(item[2])
        raise TypeError(f"Unsupported hierarchical_leiden output item: {item!r}")

    @staticmethod
    def _same_cluster(left: str, right: str, membership: dict[str, int]) -> bool:
        if left not in membership or right not in membership:
            return True
        return membership[left] == membership[right]

    @staticmethod
    def _same_type(left: Entity, right: Entity) -> bool:
        return left.entity_type.strip().casefold() == right.entity_type.strip().casefold()

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(re.findall(r"\w+", value.casefold(), flags=re.UNICODE))

    @staticmethod
    def _pair(left: str, right: str) -> tuple[str, str]:
        return (left, right) if left < right else (right, left)

    def _ngrams(self, value: str) -> set[str]:
        if not value:
            return set()
        if len(value) <= self.ngram_size:
            return {value}
        return {value[index:index + self.ngram_size] for index in range(len(value) - self.ngram_size + 1)}

    def _top_k(self, scores: dict[str, float], entities: list[Entity]) -> dict[str, float]:
        names = {entity.id: entity.entity_name for entity in entities}
        ranked = sorted(scores, key=lambda value: (-scores[value], names[value], value))
        return {value: scores[value] for value in ranked[:self.candidate_top_k]}
