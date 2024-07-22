import mlbol as dl


def assert_(cond, msg=""):
    if msg:
        assert cond, msg
    assert cond


def assert_equal(a, b, msg=""):
    if dl.is_tensor(a):
        b = dl.as_tensor(b, **dl.context(a))
        cond = dl.all(a == b)
    else:
        cond = a == b
    assert_(cond, msg)


def assert_almost_equal(a, b, rtol=1e-5, atol=1e-8, equal_nan=False, msg=None):
    dl.assert_allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan, err_msg=msg)
