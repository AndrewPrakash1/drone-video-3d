from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.align import apply_similarity, umeyama  # noqa: E402
from app.demo_scene import write_demo_files, reference_segment  # noqa: E402
from app.geo import geodetic_to_enu, make_origin  # noqa: E402
from app.metric import evaluate, evaluate_segment  # noqa: E402


def test_reference_length_is_20m():
    proof = evaluate_segment(reference_segment())
    assert proof["pass"] is True
    assert abs(proof["measured_m"] - 20.0) < 1e-6


def test_geodetic_roundtrip_preserves_eave():
    from app.demo_scene import ORIGIN_ALT, ORIGIN_LAT, ORIGIN_LON, building_corners_geodetic

    ref = building_corners_geodetic()
    o = make_origin(ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT)
    a = geodetic_to_enu(ref["a"]["lat"], ref["a"]["lon"], ref["a"]["height"], o)
    b = geodetic_to_enu(ref["b"]["lat"], ref["b"]["lon"], ref["b"]["height"], o)
    measured = float(((a - b) ** 2).sum() ** 0.5)
    result = evaluate(measured, 20.0, 1.0, 5.0)
    assert result["pass"] is True


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(12, 3))
    scale = 3.4
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    trans = np.array([10.0, -4.0, 2.0])
    dst = apply_similarity(src, scale, rot, trans)
    got = umeyama(src, dst)
    assert got is not None
    s, r, t = got
    assert abs(s - scale) < 1e-6
    assert np.allclose(r, rot, atol=1e-6)
    assert np.allclose(t, trans, atol=1e-6)


if __name__ == "__main__":
    write_demo_files(Path(__file__).resolve().parents[2] / "data" / "demo")
    test_reference_length_is_20m()
    test_geodetic_roundtrip_preserves_eave()
    test_umeyama_recovers_similarity()
    print("demo files written; metric + alignment self-check passed")
