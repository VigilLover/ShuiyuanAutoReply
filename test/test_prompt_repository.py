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
            "e6c1427aefd8fca1680d9072a56139c0ca3d4b07f4796d5b22d376348d474f1a",
        )

    def test_unknown_persona_falls_back_to_wolf(self):
        bundle = FilePromptRepository().load("unknown", set())
        self.assertEqual(bundle.persona_id, "wolf_lumine")
        self.assertEqual(
            sha256(bundle.system_prompt.encode()).hexdigest(),
            "e870441375cec4fa2a9cc7833b179cfee6e5c411870bf7340412298245f5420d",
        )

    def test_older_rule_defaults_remain_recognized_for_managed_migration(self):
        known = json.loads(
            resources.files("shuiyuan_auto_reply.prompts")
            .joinpath("legacy_defaults.json")
            .read_text()
        )
        # rules v4 and v5 forum defaults for wolf_lumine
        for digest in (
            "6bdac6be83795b872f8c5759ee61dd8f884ea44ba82b52a4305a510163c432d4",
            "35dd1cd4a2640215121939e08266a97c083f2329be1cc45d78eb5fe8b5f8d731",
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
            "8c3491be5d4fd684f7ea44100841b8ccbbf044adb4d641344ff4938827b3b056",
        )
        self.assertEqual(
            sha256(multimodal.encode()).hexdigest(),
            "b674823b96cb287c93fee334fa4f1597259fdeb12985c5a3d799eb46eb30475c",
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
        self.assertIn("artifact://", prompt)
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
