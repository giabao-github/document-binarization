def test_stroke_stub():
    from binarization.enhancement.stroke import normalize_stroke
    try:
        normalize_stroke(None)
    except NotImplementedError:
        assert True
    else:
        assert False, "normalize_stroke should be a stub"
