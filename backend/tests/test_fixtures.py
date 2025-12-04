import pytest
from app.services.llm import LLMClient
from app.workflows.agents import SimpleEmbedder
from app.data.vector_store import FaissStore


@pytest.fixture
def mock_llm(monkeypatch):
    class _MockLLM(LLMClient):
        def __init__(self):
            pass

        async def generate(self, prompt: str, max_tokens: int = 512, **kwargs):
            return f"[mocked]{prompt[:20]}"

        def generate_sync(self, prompt: str, max_tokens: int = 512, **kwargs):
            return f"[mocked]{prompt[:20]}"

    return _MockLLM()


@pytest.fixture
def mock_vector_store():
    # simple in-memory vector store with tiny dim
    return FaissStore(dim=3)


def test_simple_embedder_dimension():
    emb = SimpleEmbedder()
    assert emb.get_sentence_embedding_dimension() == 3
