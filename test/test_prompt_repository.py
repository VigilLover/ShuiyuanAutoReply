import unittest
from hashlib import sha256

from langchain_core.prompts import ChatPromptTemplate

from shuiyuan_auto_reply.application.ports.prompt import PromptScope
from shuiyuan_auto_reply.infrastructure.prompts import FilePromptRepository


class PromptRepositoryTests(unittest.TestCase):
    def test_wolf_system_prompt_v2_snapshot(self):
        bundle = FilePromptRepository().load("wolf_lumine", set())
        self.assertEqual(
            sha256(bundle.system_prompt.encode()).hexdigest(),
            "6faf3bf490358b351cf8131aef2c63734dbeac9f9cc49432a522a3bdd83e372a",
        )

    def test_unknown_persona_falls_back_to_wolf(self):
        bundle = FilePromptRepository().load("unknown", set())
        self.assertEqual(bundle.persona_id, "wolf_lumine")
        self.assertEqual(
            sha256(bundle.system_prompt.encode()).hexdigest(),
            "e4a0fe06b8a63872747bec51b51f538f43854b5af88e7d5b6b12d1d3248e1f4d",
        )

    def test_archive_and_multimodal_v2_snapshots(self):
        repository = FilePromptRepository()
        archive = repository.load("存档读取", set()).system_prompt
        multimodal = repository.load("wolf_lumine", {"multimodal"}).system_prompt
        self.assertEqual(
            sha256(archive.encode()).hexdigest(),
            "95b8b2902c74eaf0645afc75c1ef7761e8171a58f4eb0921a5e9b67ece26d392",
        )
        self.assertEqual(
            sha256(multimodal.encode()).hexdigest(),
            "9b0674dd47c6d78ec25c5246693be9f08ee13a4e494efb4425f053ec68cdbd6c",
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
