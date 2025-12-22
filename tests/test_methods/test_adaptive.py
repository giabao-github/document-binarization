def test_sauvola_stub():
    from binarization.methods.adaptive_methods import sauvola
    try:
        sauvola(None)
    except NotImplementedError:
        assert True
    else:
        assert False, "sauvola should be a stub"
