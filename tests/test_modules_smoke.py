from pathlib import Path
import json
import os
import pytest

from model_router import profile_for_role, resolve_model
from guardrails import is_high_risk, filter_tools
from knowledge_store import store_claim, CLAIMS_FILE


def test_profile_for_role_mapping():
    assert profile_for_role('Analyst') == 'brain'
    assert profile_for_role('Experimenter') in ('code','brain','fast')


def test_guardrails_filter():
    tools = ['web_search_tool','fetch_and_parse_url','search_vector']
    filtered = filter_tools(tools, allow_web=False)
    assert 'web_search_tool' not in filtered and 'fetch_and_parse_url' not in filtered


def test_is_high_risk_detects_keyword():
    assert is_high_risk('bio weaponization')


def test_store_claim_writes(tmp_path, monkeypatch):
    monkeypatch.setattr('knowledge_store.DATA_DIR', tmp_path)
    monkeypatch.setattr('knowledge_store.CLAIMS_FILE', tmp_path / 'claims.jsonl')
    store_claim('claim1', ['ev1'])
    data = (tmp_path / 'claims.jsonl').read_text(encoding='utf-8').strip().split('\n')
    payload = json.loads(data[0])
    assert payload['claim'] == 'claim1'
    assert payload['evidence_ids'] == ['ev1']
