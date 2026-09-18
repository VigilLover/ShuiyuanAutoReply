import json
import unittest
from hashlib import sha256
from importlib import resources

from langchain_core.prompts import ChatPromptTemplate

from shuiyuan_auto_reply.application.ports.prompt import PromptScope
from shuiyuan_auto_reply.infrastructure.prompts import FilePromptRepository


class PromptRepositoryTests(unittest.TestCase):
    # Hashes are deliberately frozen: changing a policy template must update them
    # by hand so a prompt edit is never an accident. Last revised with the tool-set
    # result-only final responses (rules version 5).
    def test_wolf_system_prompt_v2_snapshot(self):
        bundle = FilePromptRepository().load("wolf_lumine", set())
        self.assertEqual(
            sha256(bundle.system_prompt.encode()).hexdigest(),
            "35dd1cd4a2640215121939e08266a97c083f2329be1cc45d78eb5fe8b5f8d731",
        )

    def test_unknown_persona_falls_back_to_wolf(self):
        bundle = FilePromptRepository().load("unknown", set())
        self.assertEqual(bundle.persona_id, "wolf_lumine")
        self.assertEqual(
            sha256(bundle.system_prompt.encode()).hexdigest(),
            "f8c879d87141f23ac786cfaacdcb09a52c80b99dfbf5ed89f50958f1428e1d8b",
        )

    def test_rules_v4_defaults_remain_recognized_for_managed_migration(self):
        known = json.loads(
            resources.files("shuiyuan_auto_reply.prompts")
            .joinpath("legacy_defaults.json")
            .read_text()
        )
        self.assertEqual(
            known["6bdac6be83795b872f8c5759ee61dd8f884ea44ba82b52a4305a510163c432d4"],
            {"persona_id": "wolf_lumine", "scope": "forum"},
        )

    def test_archive_and_multimodal_v2_snapshots(self):
        repository = FilePromptRepository()
        archive = repository.load("存档读取", set()).system_prompt
        multimodal = repository.load("wolf_lumine", {"multimodal"}).system_prompt
        self.assertEqual(
            sha256(archive.encode()).hexdigest(),
            "690b8203c3ebd94b75660dac745c526ed8dde2505b7fa0ffc93d538905c40ad1",
        )
        self.assertEqual(
            sha256(multimodal.encode()).hexdigest(),
            "09885c1d9bcae4af4e4ce16d10a782f8991e69ddbf7b3404f29cb0085e3e6436",
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
