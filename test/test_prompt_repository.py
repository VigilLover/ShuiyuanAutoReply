import unittest
from hashlib import sha256

from langchain_core.prompts import ChatPromptTemplate

from shuiyuan_auto_reply.application.ports.prompt import PromptScope
from shuiyuan_auto_reply.infrastructure.prompts import FilePromptRepository


class PromptRepositoryTests(unittest.TestCase):
    def test_wolf_system_prompt_is_byte_for_byte_equivalent(self):
        bundle = FilePromptRepository().load("wolf_lumine", set())
        self.assertEqual(
            sha256(bundle.system_prompt.encode()).hexdigest(),
            "6de67ea61ab0c9729afe42cc0b52446da582aee15be2188b7daf55f7fc47f2dd",
        )

    def test_unknown_persona_falls_back_to_wolf(self):
        bundle = FilePromptRepository().load("unknown", set())
        self.assertEqual(bundle.persona_id, "wolf_lumine")
        self.assertEqual(
            sha256(bundle.system_prompt.encode()).hexdigest(),
            "dd9e43f1fb07b23b86cd9c92f83c3428a68450fb88724693533745ca4b4ade2e",
        )

    def test_archive_and_multimodal_snapshots_are_unchanged(self):
        repository = FilePromptRepository()
        archive = repository.load("存档读取", set()).system_prompt
        multimodal = repository.load("wolf_lumine", {"multimodal"}).system_prompt
        self.assertEqual(
            sha256(archive.encode()).hexdigest(),
            "63a15c0714c544d35d7aa27c60f8e405ce0d4bc8c0b01ab3f42afd1edb80ab50",
        )
        self.assertEqual(
            sha256(multimodal.encode()).hexdigest(),
            "73333d78bbd1e6b06748c78546b5cb9edcbc34b6fd5967b267c9131c9f741f1a",
        )

    def test_web_prompt_keeps_shared_rules_without_forum_write_capabilities(self):
        prompt = FilePromptRepository().load(
            "wolf_lumine", set(), PromptScope.WEB
        ).system_prompt
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
