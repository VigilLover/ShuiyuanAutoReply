import unittest
from hashlib import sha256

from langchain_core.prompts import ChatPromptTemplate

from shuiyuan_auto_reply.application.ports.prompt import PromptScope
from shuiyuan_auto_reply.infrastructure.prompts import FilePromptRepository


class PromptRepositoryTests(unittest.TestCase):
    # Hashes are deliberately frozen: changing a policy template must update them
    # by hand so a prompt edit is never an accident. Last revised with the tool-set
    # cleanup (rules version 4).
    def test_wolf_system_prompt_v2_snapshot(self):
        bundle = FilePromptRepository().load("wolf_lumine", set())
        self.assertEqual(
            sha256(bundle.system_prompt.encode()).hexdigest(),
            "f0d688a66f3e88ec86504fb991edb9bc3fb426575223577ad0f7856d06b0162c",
        )

    def test_unknown_persona_falls_back_to_wolf(self):
        bundle = FilePromptRepository().load("unknown", set())
        self.assertEqual(bundle.persona_id, "wolf_lumine")
        self.assertEqual(
            sha256(bundle.system_prompt.encode()).hexdigest(),
            "f8670657f756a336be0e0f6483520af3eddf45e5333e9b367360b35da291896b",
        )

    def test_archive_and_multimodal_v2_snapshots(self):
        repository = FilePromptRepository()
        archive = repository.load("存档读取", set()).system_prompt
        multimodal = repository.load("wolf_lumine", {"multimodal"}).system_prompt
        self.assertEqual(
            sha256(archive.encode()).hexdigest(),
            "07b9c88d4cb35888048baca3bfe14e41ec8f0d3c0acbdd7881ae9db5c8c274fa",
        )
        self.assertEqual(
            sha256(multimodal.encode()).hexdigest(),
            "bace2ee7ccb4f474f03e2de5fc3d9010cf82762a148ed03a15a010843b27835e",
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
