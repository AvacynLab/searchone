import json
from pathlib import Path
import pytest
from app.services.ingest import chunk_text


DATA_FILE = Path(__file__).parent / "data" / "sample_docs.jsonl"


def load_dataset():
    docs = []
    for line in DATA_FILE.read_text(encoding="utf-8").splitlines():
        docs.append(json.loads(line))
    return docs


def test_dataset_deterministic_chunking():
    docs = load_dataset()
    texts = [d["text"] for d in docs]
    chunks = [chunk_text(t, chunk_size=10, overlap=2) for t in texts]
    # Ensure deterministic chunk counts
    assert [len(c) for c in chunks] == [1, 1, 1]
    # Ensure content consistency
    assert "Alpha beta gamma" in chunks[0][0]


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_dataset_titles(idx):
    docs = load_dataset()
    assert docs[idx]["title"]
