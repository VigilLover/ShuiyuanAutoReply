import json
import os
import sys
import tempfile
import unittest
import logging
from pathlib import Path
from unittest.mock import patch

import dotenv
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shuiyuan_auto_reply.features.mention.mention_pet_model import MentionPetModel


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


class TestMentionPetModel(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        dotenv.load_dotenv()
        cls.api_key = os.getenv("DEEPSEEK_API_KEY")

    def setUp(self):
        logging.info("[SETUP] 创建临时测试目录和测试资产文件")
        self.model = None
        self.tmpdir = tempfile.TemporaryDirectory()
        root = Path(self.tmpdir.name)

        self.responses_path = root / "pet_responses.json"
        self.state_path = root / "pet_state.json"
        self.endings_path = root / "pet_endings.json"

        responses = {
            "开心": {
                "weight": 10,
                "texts": ["本地默认文案"],
                "ascii": ["owo"],
                "deltas": {
                    "patience": 10,
                    "wisdom": -5,
                    "chaos": 3,
                },
                "special_overrides": {
                    "special_user": {
                        "texts": ["特殊用户文案"],
                        "ascii": ["uwu"],
                        "deltas": {
                            "patience": 20,
                            "wisdom": 1,
                            "chaos": -2,
                        },
                    }
                },
            }
        }

        endings = {
            "patience_max": {
                "title": "【隐藏结局：圣狼降临】",
                "text": "达成耐心满值结局",
                "ascii": "END",
            }
        }

        self.responses_path.write_text(json.dumps(responses, ensure_ascii=False), encoding="utf-8")
        self.endings_path.write_text(json.dumps(endings, ensure_ascii=False), encoding="utf-8")
        logging.info("[SETUP] 测试资产写入完成: responses=%s, endings=%s", self.responses_path, self.endings_path)

    def tearDown(self):
        logging.info("[TEARDOWN] 清理临时目录")
        self.tmpdir.cleanup()

    def _build_model(self, *, live: bool = False) -> MentionPetModel:
        # 允许真实 LLM，同时禁用 retriever 初始化以避免外部 Neo4j 依赖干扰测试
        logging.info("[BUILD] 构建 MentionPetModel 实例")
        if live and not self.api_key:
            self.skipTest("DEEPSEEK_API_KEY not set")
        key = self.api_key if live else "offline-test-key"
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": key}):
            model = MentionPetModel(
                filepath=str(self.responses_path),
                state_path=str(self.state_path),
                endings_path=str(self.endings_path),
                persona="wolf_lumine",
            )
        model.retriever = None
        self.model = model
        logging.info("[BUILD] 模型构建完成，retriever_enabled=%s", bool(model.retriever))
        return model

    async def asyncTearDown(self):
        if self.model is not None:
            await self.model.client.close()

    async def test_get_rua_response_without_user_text_uses_local_text(self):
        logging.info("[CASE] 开始: 空 user_text 时使用本地文案")
        model = self._build_model()

        logging.info("[STEP] 调用 get_rua_response(user_text='')")
        with patch("shuiyuan_auto_reply.features.mention.mention_pet_model.random.randint", return_value=0):
            reply = await model.get_rua_response(username="normal_user", name="normal_name", user_text="")
        logging.info("[STEP] 模型回复: %s", reply)

        self.assertIsNotNone(reply)
        self.assertIn("【开心】 本地默认文案", reply)
        self.assertIn("耐心", reply)
        self.assertIn("智慧", reply)
        self.assertIn("混沌", reply)

        saved_state = json.loads(self.state_path.read_text(encoding="utf-8"))
        logging.info("[STEP] 保存状态: %s", saved_state)
        self.assertEqual(saved_state["patience"], 10)
        self.assertEqual(saved_state["wisdom"], -5)
        self.assertEqual(saved_state["chaos"], 3)
        logging.info("[CASE] 通过: 空 user_text 本地文案与状态更新正确")

    @pytest.mark.live
    async def test_get_rua_response_with_user_text_uses_real_llm_text(self):
        logging.info("[CASE] 开始: 有 user_text 时真实调用大模型生成文案")
        model = self._build_model(live=True)

        logging.info("[STEP] 调用 get_rua_response(user_text='今天有点累，但还是想被摸摸头')")
        with patch("shuiyuan_auto_reply.features.mention.mention_pet_model.random.randint", return_value=0):
            reply = await model.get_rua_response(
                username="normal_user",
                name="normal_name",
                user_text="今天有点累，但还是想被摸摸头",
            )
        logging.info("[STEP] 模型回复: %s", reply)

        self.assertIsNotNone(reply)
        self.assertIn("【开心】 ", reply)
        self.assertNotIn("【开心】 本地默认文案", reply)
        logging.info("[CASE] 通过: 真实 LLM 文案已写入最终回复")

    @pytest.mark.live
    async def test_generate_personalized_text_direct_real_call(self):
        logging.info("[CASE] 开始: 直接调用 _generate_personalized_text 进行真实模型验证")
        model = self._build_model(live=True)

        logging.info("[STEP] 调用 _generate_personalized_text")
        text = await model._generate_personalized_text(
            user_text="刚考完试有点紧张",
            selected_state="开心",
            deltas={"patience": 10, "wisdom": -5, "chaos": 3},
            state={"patience": 10, "wisdom": -5, "chaos": 3},
        )
        logging.info("[STEP] 生成结果: %s", text)

        self.assertIsNotNone(text)
        self.assertTrue(text.strip())
        self.assertLessEqual(len(text), 80)
        logging.info("[CASE] 通过: 真实大模型成功返回个性化短句")

    async def test_special_override_applies_for_specific_user(self):
        logging.info("[CASE] 开始: special_overrides 对特定用户生效")
        model = self._build_model()

        logging.info("[STEP] 调用 get_rua_response(username='special_user')")
        with patch("shuiyuan_auto_reply.features.mention.mention_pet_model.random.randint", return_value=0):
            reply = await model.get_rua_response(username="special_user", name="special_user", user_text="")
        logging.info("[STEP] 模型回复: %s", reply)

        self.assertIsNotNone(reply)
        self.assertIn("【开心】 特殊用户文案", reply)

        saved_state = json.loads(self.state_path.read_text(encoding="utf-8"))
        logging.info("[STEP] 保存状态: %s", saved_state)
        self.assertEqual(saved_state["patience"], 20)
        self.assertEqual(saved_state["wisdom"], 1)
        self.assertEqual(saved_state["chaos"], -2)
        logging.info("[CASE] 通过: 覆盖文案和覆盖属性生效")

    async def test_ending_trigger_resets_state(self):
        logging.info("[CASE] 开始: 结局触发后状态重置")
        model = self._build_model()
        logging.info("[STEP] 预置状态为 patience=95 以触发 patience_max")
        self.state_path.write_text(
            json.dumps({"patience": 95, "wisdom": 0, "chaos": 0}, ensure_ascii=False),
            encoding="utf-8",
        )

        logging.info("[STEP] 调用 get_rua_response 触发结局")
        with patch("shuiyuan_auto_reply.features.mention.mention_pet_model.random.randint", return_value=0):
            reply = await model.get_rua_response(username="normal_user", name="normal_name", user_text="")
        logging.info("[STEP] 模型回复: %s", reply)

        self.assertIsNotNone(reply)
        self.assertIn("【隐藏结局：圣狼降临】", reply)
        self.assertIn("所有属性重新归零", reply)

        saved_state = json.loads(self.state_path.read_text(encoding="utf-8"))
        logging.info("[STEP] 保存状态: %s", saved_state)
        self.assertEqual(saved_state, {"patience": 0, "wisdom": 0, "chaos": 0})
        logging.info("[CASE] 通过: 结局触发与状态归零正确")


if __name__ == "__main__":
    unittest.main(verbosity=2)
