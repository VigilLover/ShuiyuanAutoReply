import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SIMPLEMCP_REVISION = "2ab491f5fe4c9a7cec8bf44bafa3363676e31a96"


def _prepare_sources_module():
    path = ROOT / "scripts/deploy/prepare_sources.py"
    spec = importlib.util.spec_from_file_location("prepare_sources", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_simplemcp_build_input_is_pinned_to_structured_web_revision():
    module = _prepare_sources_module()

    assert module.SOURCES["simplemcp"][1] == EXPECTED_SIMPLEMCP_REVISION


def test_mcp_image_runs_the_canonical_vendored_server():
    dockerfile = (ROOT / "deploy/mcp/Dockerfile").read_text()

    assert "COPY deploy/vendor/simplemcp /srv/mcp" in dockerfile
    assert "WORKDIR /srv/mcp" in dockerfile
    assert (
        'CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "58000"]'
        in dockerfile
    )
    assert "deploy/mcp/fetch.py" not in dockerfile
    assert "deploy/mcp/server.py" not in dockerfile


def test_no_production_fetch_overlay_remains():
    assert not (ROOT / "deploy/mcp/fetch.py").exists()
    assert not (ROOT / "deploy/mcp/server.py").exists()
