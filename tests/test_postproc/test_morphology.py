def test_morphology_stub():
    from binarization.postproc.morphology import opening
    try:
        opening(None)
    except NotImplementedError:
        assert True
    else:
        assert False, "opening should be a stub"
