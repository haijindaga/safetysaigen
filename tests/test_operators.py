import numpy as np

from core_safety.predicates import Predicate, SafetyConstraints
from core_safety.grounding.operators import predicate_to_mask
from core_safety.grounding.image_safe_set import build_image_safe_set


def _seg_with_blob():
    seg = {"cone": np.zeros((100, 200), dtype=bool),
           "floor": np.zeros((100, 200), dtype=bool)}
    seg["cone"][40:50, 20:30] = True     # instance 1
    seg["cone"][40:50, 120:130] = True   # instance 2
    seg["floor"][60:, :] = True
    return seg


def test_on_near_are_class_pixels():
    seg = _seg_with_blob()
    on = predicate_to_mask(Predicate("ON", "floor"), seg)
    assert (on == seg["floor"]).all()
    near = predicate_to_mask(Predicate("NEAR", "cone"), seg)
    assert (near == seg["cone"]).all()


def test_around_dilates():
    seg = _seg_with_blob()
    around = predicate_to_mask(Predicate("AROUND", "cone"), seg, around_kernel_px=21)
    assert around.sum() > seg["cone"].sum()
    assert around[seg["cone"]].all()          # superset
    assert around[45, 35]                     # ~10 px beyond the instance


def test_between_fills_gap():
    seg = _seg_with_blob()
    between = predicate_to_mask(Predicate("BETWEEN", "cone"), seg)
    # Convex hull covers the gap between the two instances.
    assert between[45, 75]
    assert not between[10, 75]


def test_missing_class_is_empty():
    seg = _seg_with_blob()
    mask = predicate_to_mask(Predicate("NEAR", "ghost"), seg)
    assert mask.shape == (100, 200) and not mask.any()


def test_image_safe_set_partition():
    seg = _seg_with_blob()
    c = SafetyConstraints(
        safe=[Predicate("ON", "floor")],
        unsafe=[Predicate("BETWEEN", "cone")])
    safe, unsafe = build_image_safe_set(c, seg)
    assert not (safe & unsafe).any()          # Eq. 2: unsafe subtracted
    # Floor pixels inside the hull are unsafe, outside remain safe.
    assert unsafe[45, 75]
    assert safe[90, 5]
