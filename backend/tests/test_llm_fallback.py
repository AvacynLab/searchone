import asyncio
from app.services.llm import LLMClient


def test_llm_fallback_local_when_no_key():
    llm = LLMClient(api_key=None, fallback_local=True, use_sdk=False)
    out = asyncio.run(llm.generate("hello", max_tokens=10))
    assert out["text"].startswith("[local-fallback]")


def test_llm_fallback_sync():
    llm = LLMClient(api_key=None, fallback_local=True, use_sdk=False)
    out = llm.generate_sync("hello", max_tokens=10)
    assert out.startswith("[local-fallback]")
