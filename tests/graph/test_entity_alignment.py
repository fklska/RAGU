from unittest.mock import AsyncMock, MagicMock, patch

from ragu.chunker.types import Chunk
from ragu.graph.artifacts_summarizer import RelationSummarizer
from ragu.graph.entity_alignment import EntityAligner
from ragu.graph.graph_builder_pipeline import (
    BuilderArguments,
    GraphBuilderModule,
    InMemoryGraphBuilder,
)
from ragu.graph.knowledge_graph import KnowledgeGraph
from ragu.graph.types import Entity, Relation
from ragu.models.decisions import (
    EntityDecider,
    EntityDecision,
    EntityDecisionBatch,
    LLMEntityAlignmentDecider,
)
from ragu.models.embedder import Embedder
from ragu.models.llm import LLM


class DummyEmbedder(Embedder):
    def __init__(self, vectors_by_name=None, default_vector=None):
        self.vectors_by_name = vectors_by_name or {}
        self.default_vector = default_vector or [0.0, 0.0]
        self.calls = []

    @property
    def dim(self):
        return len(self.default_vector)

    def vector(self, text):
        name = text.split(" - ", 1)[0]
        return self.vectors_by_name.get(name, self.default_vector)

    async def embed_text(self, text, **kwargs):
        return self.vector(text)

    async def batch_embed_text(self, texts, desc=None, **kwargs):
        self.calls.append(list(texts))
        return [self.vector(text) for text in texts]


class RecordingDecider(EntityDecider):
    def __init__(self, accepted=None):
        self.accepted = {frozenset(pair) for pair in accepted or []}
        self.calls = []

    async def decide(self, entity, candidates):
        self.calls.append((entity, list(candidates)))
        return [
            EntityDecision(
                candidate_id=candidate.id,
                same_entity=frozenset((entity.entity_name, candidate.entity_name)) in self.accepted,
            )
            for candidate in candidates
        ]


def make_entity(name, entity_type="Person", description=None, chunks=None, documents=None):
    return Entity(
        entity_name=name,
        entity_type=entity_type,
        description=description or f"Description of {name}",
        source_chunk_id=chunks or [f"chunk-{name}"],
        documents_id=documents or [],
    )


def make_relation(subject, object_, chunks=None, description="knows", strength=1.0):
    return Relation(
        subject_id=subject.id,
        object_id=object_.id,
        subject_name=subject.entity_name,
        object_name=object_.entity_name,
        relation_type="KNOWS",
        description=description,
        relation_strength=strength,
        source_chunk_id=chunks or [],
    )


def candidates_for(decider, name):
    for entity, candidates in decider.calls:
        if entity.entity_name == name:
            return [candidate.entity_name for candidate in candidates]
    return []


async def test_ngram_candidates_exclude_self_and_other_types():
    entities = [
        make_entity("Ivan Petrov"),
        make_entity("I. Petrov"),
        make_entity("Ivan Petrov", entity_type="Organization"),
    ]
    decider = RecordingDecider()
    aligner = EntityAligner(DummyEmbedder(), decider)

    aligned, relations = await aligner.run(entities, [])

    assert aligned == entities
    assert relations == []
    assert candidates_for(decider, "Ivan Petrov") == ["I. Petrov"]
    assert candidates_for(decider, "I. Petrov") == ["Ivan Petrov"]
    assert aligner.last_report.candidate_pairs == 1
    assert aligner.last_report.accepted_pairs == 0


async def test_dense_candidates_and_candidate_limit():
    entities = [make_entity("a"), make_entity("b"), make_entity("c")]
    embedder = DummyEmbedder({
        "a": [1.0, 0.0],
        "b": [0.9, 0.1],
        "c": [0.0, 1.0],
    })
    decider = RecordingDecider()
    aligner = EntityAligner(
        embedder,
        decider,
        candidate_top_k=1,
        max_candidates_per_entity=1,
    )

    await aligner.run(entities, [])

    assert candidates_for(decider, "a") == ["b"]
    assert all(len(candidates) <= 1 for _, candidates in decider.calls)


async def test_merge_preserves_provenance_and_rewrites_relations():
    alice = make_entity(
        "Alice",
        description="Researcher.",
        chunks=["chunk-1"],
        documents=["doc-1"],
    )
    alice_smith = make_entity(
        "Alice Smith",
        description="Researcher. Scientist.",
        chunks=["chunk-2"],
        documents=["doc-2"],
    )
    bob = make_entity("Bob")
    relations = [
        make_relation(alice, bob, ["chunk-1"], strength=1.0),
        make_relation(alice_smith, bob, ["chunk-2"], strength=3.0),
        make_relation(alice, alice_smith),
    ]
    decider = RecordingDecider(accepted=[("Alice", "Alice Smith")])
    aligner = EntityAligner(DummyEmbedder(), decider)

    entities, rewritten = await aligner.run([alice, alice_smith, bob], relations)
    summarized = await RelationSummarizer(use_llm_summarization=False).run(rewritten)

    assert len(entities) == 2
    canonical = next(entity for entity in entities if entity.id == alice_smith.id)
    assert canonical.entity_name == "Alice Smith"
    assert canonical.description == "Researcher. Scientist."
    assert canonical.source_chunk_id == ["chunk-1", "chunk-2"]
    assert canonical.documents_id == ["doc-1", "doc-2"]
    assert len(summarized) == 1
    relation = summarized[0]
    assert relation.subject_id == alice_smith.id
    assert relation.object_id == bob.id
    assert relation.subject_name == "Alice Smith"
    assert relation.object_name == "Bob"
    assert sorted(relation.source_chunk_id) == ["chunk-1", "chunk-2"]
    assert relation.relation_strength == 2.0
    assert relation.id != relations[0].id
    assert relation.subject_id in {entity.id for entity in entities}
    assert relation.object_id in {entity.id for entity in entities}
    assert aligner.last_report.rewritten_relations == 1
    assert aligner.last_report.dropped_self_loops == 1
    assert aligner.last_report.merged_components == 1


async def test_transitive_acceptance_merges_component():
    entities = [make_entity("a"), make_entity("b"), make_entity("c")]
    decider = RecordingDecider(accepted=[("a", "b"), ("b", "c")])
    aligner = EntityAligner(DummyEmbedder(default_vector=[1.0, 0.0]), decider)

    merged, _ = await aligner.run(entities, [])

    assert len(merged) == 1
    assert merged[0].entity_name == "a"
    assert aligner.last_report.accepted_pairs == 2
    assert aligner.last_report.merged_components == 1


async def test_alignment_is_deterministic_for_input_permutation():
    entities = [
        make_entity("Alice", description="Researcher.", chunks=["chunk-1"]),
        make_entity("Alice Smith", description="Scientist.", chunks=["chunk-2"]),
        make_entity("Bob"),
    ]
    accepted = [("Alice", "Alice Smith")]

    async def align(values):
        decider = RecordingDecider(accepted=accepted)
        aligner = EntityAligner(DummyEmbedder(), decider)
        return await aligner.run(values, [])

    first, _ = await align(entities)
    second, _ = await align(list(reversed(entities)))

    first_by_id = {entity.id: entity for entity in first}
    second_by_id = {entity.id: entity for entity in second}
    assert first_by_id.keys() == second_by_id.keys()
    for entity_id, first_entity in first_by_id.items():
        assert first_entity.description == second_by_id[entity_id].description
        assert first_entity.source_chunk_id == second_by_id[entity_id].source_chunk_id


async def test_ngram_state_does_not_leak_between_runs():
    decider = RecordingDecider()
    aligner = EntityAligner(DummyEmbedder(), decider)
    await aligner.run([make_entity("Ivan Petrov"), make_entity("I. Petrov")], [])
    calls_after_first_run = len(decider.calls)

    entities, _ = await aligner.run([make_entity("Unrelated"), make_entity("Quantum")], [])

    assert len(entities) == 2
    assert len(decider.calls) == calls_after_first_run
    assert aligner.last_report.candidate_pairs == 0
    assert not hasattr(aligner, "ngram_index")


async def test_single_entity_skips_embedding_and_decider():
    embedder = DummyEmbedder()
    decider = RecordingDecider()
    aligner = EntityAligner(embedder, decider)
    entity = make_entity("Isolated")

    entities, relations = await aligner.run([entity], [])

    assert entities == [entity]
    assert relations == []
    assert embedder.calls == []
    assert decider.calls == []
    assert aligner.last_report.entities_after == 1


async def test_leiden_blocks_dense_candidates_with_isolated_fallback(monkeypatch):
    entities = [make_entity(name) for name in ("a", "b", "c", "d")]
    by_name = {entity.entity_name: entity for entity in entities}
    relations = [
        make_relation(by_name["a"], by_name["b"]),
        make_relation(by_name["b"], by_name["c"]),
    ]
    mapping = [
        {"node": by_name["a"].id, "cluster": 1, "level": 0},
        {"node": by_name["b"].id, "cluster": 1, "level": 0},
        {"node": by_name["c"].id, "cluster": 2, "level": 0},
    ]
    monkeypatch.setattr(
        "ragu.graph.entity_alignment.hierarchical_leiden",
        lambda *args, **kwargs: mapping,
    )
    decider = RecordingDecider()
    aligner = EntityAligner(
        DummyEmbedder(default_vector=[1.0, 0.0]),
        decider,
        cluster_scope="dense",
    )

    await aligner.run(entities, relations)

    assert set(candidates_for(decider, "a")) == {"b", "d"}
    assert "c" not in candidates_for(decider, "a")
    assert set(candidates_for(decider, "d")) == {"a", "b", "c"}
    assert all(entity.clusters == [] for entity in entities)


async def test_pipeline_aligns_before_relation_summarizer_and_communities():
    original_entities = [make_entity("Alice"), make_entity("Bob")]
    original_relations = [make_relation(original_entities[0], original_entities[1])]
    extractor = AsyncMock(return_value=(original_entities, original_relations))
    builder = InMemoryGraphBuilder(
        embedder=DummyEmbedder(),
        llm=AsyncMock(spec=LLM),
        artifact_extractor=extractor,
        build_parameters=BuilderArguments(
            use_llm_summarization=False,
            make_community_summary=True,
            remove_isolated_nodes=False,
        ),
        entity_aligner=AsyncMock(spec=GraphBuilderModule),
    )
    order = []

    async def summarize_entities(entities):
        order.append("entities")
        return entities

    async def align(entities, relations):
        order.append("align")
        return [entities[0]], []

    async def summarize_relations(relations):
        order.append("relations")
        return relations

    async def nearby(chunks, entities, relations):
        order.append("nearby")
        return [], []

    async def cluster(entities, relations):
        order.append("cluster")
        assert entities == [original_entities[0]]
        assert relations == []
        return []

    class AdditionalModule(GraphBuilderModule):
        async def run(self, entities, relations, **kwargs):
            order.append("additional")
            return entities, relations

    builder.entity_summarizer.run = AsyncMock(side_effect=summarize_entities)
    builder.entity_aligner.run = AsyncMock(side_effect=align)
    builder.relation_summarizer.run = AsyncMock(side_effect=summarize_relations)
    builder.nearby_relation_extractor.a_extract_nearby = AsyncMock(side_effect=nearby)
    builder.additional_pipeline = [AdditionalModule()]
    builder.community_summarizer = AsyncMock()

    with patch.object(builder, "cluster_graph", new=AsyncMock(side_effect=cluster)):
        entities, relations, _, _, _ = await builder.extract_graph([
            Chunk(content="text", chunk_order_idx=0, doc_id="doc-1")
        ])

    assert entities == [original_entities[0]]
    assert relations == []
    assert order == ["entities", "align", "relations", "nearby", "additional", "cluster"]


async def test_builder_without_aligner_keeps_default_behavior():
    builder = InMemoryGraphBuilder(
        embedder=DummyEmbedder(),
        build_parameters=BuilderArguments(build_only_vector_context=True),
    )
    chunk = Chunk(content="text", chunk_order_idx=0, doc_id="doc-1")

    result = await builder.extract_graph([chunk])

    assert builder.entity_aligner is None
    assert result == ([], [], [], [], [chunk])


def test_knowledge_graph_passes_aligner_to_pipeline():
    aligner = AsyncMock(spec=GraphBuilderModule)
    with patch("ragu.graph.knowledge_graph.Index", return_value=MagicMock()):
        graph = KnowledgeGraph(
            llm=None,
            embedder=DummyEmbedder(),
            builder_settings=BuilderArguments(
                use_llm_summarization=False,
                make_community_summary=False,
                remove_isolated_nodes=False,
            ),
            entity_aligner=aligner,
        )

    assert graph.pipeline.entity_aligner is aligner


async def test_llm_decider_uses_structured_output():
    source = make_entity("Ivan Petrov")
    candidate = make_entity("I. Petrov")
    llm = AsyncMock(spec=LLM)
    llm.chat_completion.return_value = EntityDecisionBatch(
        decisions=[EntityDecision(candidate_id=candidate.id, same_entity=True)]
    )
    decider = LLMEntityAlignmentDecider(llm)

    decisions = await decider.decide(source, [candidate])

    assert decisions[0].candidate_id == candidate.id
    assert decisions[0].same_entity is True
    kwargs = llm.chat_completion.await_args.kwargs
    assert kwargs["output_schema"] is EntityDecisionBatch
    assert candidate.id in str(kwargs["conversation"])
