def test_otsu_stub():
    from binarization.methods.global_methods import otsu
    try:
        otsu(None)
    except NotImplementedError:
        assert True
    else:
        assert False, "otsu should be a stub"
