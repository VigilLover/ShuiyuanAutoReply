import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "src" / "shuiyuan_auto_reply"


def project_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


class ArchitectureBoundaryTests(unittest.TestCase):
    def test_application_does_not_import_frameworks_or_infrastructure(self):
        forbidden = (
            "aiohttp",
            "fastapi",
            "sqlalchemy",
            "neo4j",
            "neomodel",
            "shuiyuan_auto_reply.infrastructure",
        )
        offenders = []
        for path in (ROOT / "application").rglob("*.py"):
            for imported in project_imports(path):
                if imported.startswith(forbidden):
                    offenders.append(f"{path.relative_to(ROOT)} -> {imported}")
        self.assertEqual(offenders, [])

    def test_domain_does_not_import_project_frameworks(self):
        allowed = {"dataclasses", "enum", "conversation", "message", "response"}
        offenders = []
        for path in (ROOT / "domain").rglob("*.py"):
            for imported in project_imports(path):
                root = imported.split(".", 1)[0]
                if root not in allowed and not imported.startswith("domain"):
                    offenders.append(f"{path.relative_to(ROOT)} -> {imported}")
        self.assertEqual(offenders, [])

    def test_production_does_not_import_examples(self):
        offenders = []
        for path in ROOT.rglob("*.py"):
            if any(name.startswith("examples") for name in project_imports(path)):
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])
