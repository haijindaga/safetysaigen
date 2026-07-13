import pytest

from core_safety.predicates import Predicate, parse_predicates, parse_vlm_output


def test_parse_predicates():
    preds = parse_predicates("NEAR(desk), AROUND(wet_floor_sign), BETWEEN(cone)")
    assert preds == [Predicate("NEAR", "desk"),
                     Predicate("AROUND", "wet_floor_sign"),
                     Predicate("BETWEEN", "cone")]


def test_parse_case_and_spaces():
    preds = parse_predicates("near( Traffic Cone )")
    assert preds == [Predicate("NEAR", "traffic_cone")]


def test_parse_vlm_output_with_fences_and_trailing_comma():
    text = """Here is the analysis:
```json
{
"safety_logic": "cones form a line",
"classes": "cone, floor",
"unsafe_regions": "BETWEEN(cone), NEAR(cone)",
"safe_regions": "ON(floor)",
}
```"""
    c = parse_vlm_output(text)
    assert c.safe == [Predicate("ON", "floor")]
    assert Predicate("BETWEEN", "cone") in c.unsafe
    assert set(c.all_classes()) == {"cone", "floor"}


def test_parse_vlm_output_invalid_raises():
    with pytest.raises(ValueError):
        parse_vlm_output("I think the floor is safe to traverse.")
