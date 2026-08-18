import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class EnvironmentExampleContract(unittest.TestCase):
    def test_every_directly_read_environment_variable_is_documented(self):
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "src" / "shuiyuan_auto_reply").rglob("*.py")
        )
        used = set(
            re.findall(
                r'(?:getenv|_value|_text|_flag)\(\s*["\']([A-Z][A-Z0-9_]*)',
                source,
            )
        )
        documented = set()
        for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
            if match := re.match(r"\s*#?\s*([A-Z][A-Z0-9_]*)\s*=", line):
                documented.add(match.group(1))
        self.assertEqual(sorted(used - documented), [])
