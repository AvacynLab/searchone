from app.services.nlp_structuring import extract_entities, extract_relations, normalize_entities


def test_extract_entities_and_normalization():
    text = "La supraconductivité permet aux scientifiques de travailler sur des aimants quantiques."
    entities = extract_entities(text)
    normalized = normalize_entities(entities)
    assert normalized
    for ent in normalized:
        assert ent.get("normalized") == ent.get("normalized", "").strip().lower()


def test_extract_relations_simple():
    text = "La supraconductivité cause une résistance nulle."
    entities = extract_entities(text)
    relations = extract_relations(text, entities)
    assert isinstance(relations, list)
    if relations:
        assert relations[0].get("type") in {"causes", "part_of", "uses"}
