import json
import unittest
from hashlib import sha256
from importlib import resources

from langchain_core.prompts import ChatPromptTemplate

from shuiyuan_auto_reply.application.ports.prompt import PromptScope
from shuiyuan_auto_reply.infrastructure.prompts import FilePromptRepository


class PromptRepositoryTests(unittest.TestCase):
    # Hashes are deliberately frozen: changing a policy template must update them
    # by hand so a prompt edit is never an accident. Last revised with the compact
    # v1.1.0 templates (rules version 6).
    def test_wolf_system_prompt_v2_snapshot(self):
        bundle = FilePromptRepository().load("wolf_lumine", set())
        self.assertEqual(
            sha256(bundle.system_prompt.encode()).hexdigest(),
            "aedb6289bc3c1fb309749b1e5869d881789ec08bfe89b7397f0676c443dbb211",
        )

    def test_unknown_persona_falls_back_to_wolf(self):
        bundle = FilePromptRepository().load("unknown", set())
        self.assertEqual(bundle.persona_id, "wolf_lumine")
        self.assertEqual(
            sha256(bundle.system_prompt.encode()).hexdigest(),
            "ccf8a6d93d831e2abe1f974c931e338a8c02afaee7a5fd50a01e36d99e056dbe",
        )

    def test_older_rule_defaults_remain_recognized_for_managed_migration(self):
        known = json.loads(
            resources.files("shuiyuan_auto_reply.prompts")
            .joinpath("legacy_defaults.json")
            .read_text()
        )
        # rules v4, v5 and v6 forum defaults for wolf_lumine
        for digest in (
            "6bdac6be83795b872f8c5759ee61dd8f884ea44ba82b52a4305a510163c432d4",
            "35dd1cd4a2640215121939e08266a97c083f2329be1cc45d78eb5fe8b5f8d731",
            "e6c1427aefd8fca1680d9072a56139c0ca3d4b07f4796d5b22d376348d474f1a",
            "b674823b96cb287c93fee334fa4f1597259fdeb12985c5a3d799eb46eb30475c",
        ):
            self.assertEqual(
                known[digest], {"persona_id": "wolf_lumine", "scope": "forum"}
            )

    def test_archive_and_multimodal_v2_snapshots(self):
        repository = FilePromptRepository()
        archive = repository.load("存档读取", set()).system_prompt
        multimodal = repository.load("wolf_lumine", {"multimodal"}).system_prompt
        self.assertEqual(
            sha256(archive.encode()).hexdigest(),
            "8a71b75d77a760bba3b979299e20d986ae14bbdd4b6ca0cf6fc75989859f4c4a",
        )
        self.assertEqual(
            sha256(multimodal.encode()).hexdigest(),
            "0a8b91c4c57a9460ed763e466cce014e016830d34d9fcfd944b937c6e6e7c52c",
        )

    def test_web_prompt_keeps_shared_rules_without_forum_write_capabilities(self):
        prompt = (
            FilePromptRepository()
            .load("wolf_lumine", set(), PromptScope.WEB)
            .system_prompt
        )
        self.assertIn("【安全与防御规则】", prompt)
        self.assertIn("【工具使用说明】", prompt)
        self.assertIn("【图片生成 - 严格规则】", prompt)
        self.assertIn("【长期记忆工具】", prompt)
        self.assertIn("不能创建或编辑论坛帖子", prompt)
        self.assertIn("[图N]", prompt)
        self.assertIn("#hN", prompt)
        self.assertNotIn("artifact://", prompt)
        self.assertIn("不描述查询、工具、失败、重试或核实过程", prompt)
        self.assertNotIn("最终只输出给用户【{username}】看的回帖正文", prompt)
        rendered = ChatPromptTemplate.from_template(prompt).invoke(
            {
                "user_id": "web:account-a",
                "username": "web-user",
                "name": "",
                "long_term_memory": "无相关长期记忆",
                "context": "",
            }
        )
        self.assertIn("当前网页用户 ID: web:account-a", rendered.to_string())
