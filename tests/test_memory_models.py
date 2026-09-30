"""The real embedder and semantic retrieval over a small, realistic memory."""

import numpy as np
import pytest

from kin.memory.store import Store
from kin.perception.segmenter import EpisodePolicy

pytestmark = pytest.mark.models

T0 = 1_800_000_000.0
EPISODES = [
    "Rohan said the site visit is Thursday at 4 pm at Butibori.",
    "The budget meeting moved to Friday afternoon. Priya will send the slides tomorrow morning.",
    "I had pasta for lunch and then went for a walk in the park.",
    "The landlord called about fixing the kitchen sink next week.",
    "My sister's flight lands at 9 in the evening on Sunday.",
]


@pytest.fixture(scope="module")
def embedder():
    from kin.memory.embed import Embedder

    return Embedder()


@pytest.fixture
def memory(tmp_path, embedder):
    from kin.memory.service import Memory

    mem = Memory(Store(tmp_path / "kin.db", "ab" * 32), embedder, EpisodePolicy())
    for i, text in enumerate(EPISODES):
        mem.add(T0 + i * 600, T0 + i * 600 + 5, text, is_command=False, redactions=0)
    mem.store.close_all()
    for ep in mem.store.pending_embeddings():
        assert mem.embed_episode(ep)
    yield mem
    mem.store.close()


def test_embeddings_are_unit_384d_and_stable(embedder):
    v = embedder.embed(["hello there", "hello there", "tax return"])
    assert v.shape == (3, 384) and v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-4)
    assert float(v[0] @ v[1]) > 0.999 and float(v[0] @ v[2]) < 0.5


def test_similar_meaning_scores_higher(embedder):
    q, near, far = embedder.embed(["When is the construction site inspection?", EPISODES[0], EPISODES[2]])
    assert q @ near > q @ far + 0.2


@pytest.mark.parametrize(
    "query,expected",
    [
        ("What did Priya say about the budget?", 1),  # keyword + meaning
        ("should I take an umbrella to the site visit Rohan mentioned", 0),
        ("what did I eat at midday", 2),  # meaning only: no shared keywords
        ("who needs to repair the plumbing at home", 3),  # meaning only
        ("when does the airplane arrive", 4),  # meaning only
    ],
)
def test_search_finds_the_right_episode(memory, embedder, query, expected):
    hits = memory.search(query, k=3, qvec=embedder.embed_one(query))
    assert hits[0]["text"] == EPISODES[expected], [h["text"][:40] for h in hits]


def test_meaning_only_hit_has_no_keyword_match(memory, embedder):
    q = "what did I eat at midday"
    top = memory.search(q, k=1, qvec=embedder.embed_one(q))[0]
    assert not top["keyword_match"] and top["vector_similarity"] > 0.3


def test_unrelated_question_scores_low(memory, embedder):
    q = "draft my tax return"
    hits = memory.search(q, k=3, qvec=embedder.embed_one(q))
    assert all(h["vector_similarity"] < 0.3 and not h["keyword_match"] for h in hits)
