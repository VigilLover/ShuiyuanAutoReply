import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="run tests that contact real forum/database/model services",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "live: requires explicitly enabled external services")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-live"):
        return
    skip = pytest.mark.skip(reason="live test; pass --run-live to enable")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
