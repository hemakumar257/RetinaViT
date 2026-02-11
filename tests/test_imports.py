def test_package_import():
    import importlib

    m = importlib.import_module("retinavit")
    assert hasattr(m, "__version__")
