def test_advanced_stubs():
    from binarization.methods.advanced_methods import clahe_enhance
    try:
        clahe_enhance(None)
    except NotImplementedError:
        assert True
    else:
        assert False, "clahe_enhance should be a stub"
