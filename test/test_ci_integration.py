import importlib.util
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

_PATH = Path(__file__).resolve().parents[1] / "scripts/ci/integration.py"
_SPEC = importlib.util.spec_from_file_location("ci_integration", _PATH)
assert _SPEC and _SPEC.loader
integration = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(integration)


def test_wait_for_probe_retries_during_container_restart():
    unavailable = subprocess.CalledProcessError(1, ["docker", "compose", "exec"])
    with (
        patch.object(
            integration,
            "probe",
            side_effect=[unavailable, unavailable, '{"status": "ok"}'],
        ) as probe,
        patch.object(integration.time, "sleep") as sleep,
    ):
        result = integration.wait_for_probe(
            "/api/live",
            decoder=json.loads,
            ready=lambda value: value.get("status") == "ok",
        )

    assert result == {"status": "ok"}
    assert probe.call_count == 3
    assert sleep.call_count == 2
