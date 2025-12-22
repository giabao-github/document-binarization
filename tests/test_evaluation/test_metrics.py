def test_metrics_stub():
    from binarization.evaluation.metrics import compute_metrics
    try:
        compute_metrics(None, None)
    except NotImplementedError:
        assert True
    else:
        assert False, "compute_metrics should be a stub"
