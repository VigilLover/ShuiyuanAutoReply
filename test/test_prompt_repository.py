import unittest
from hashlib import sha256

from shuiyuan_auto_reply.infrastructure.prompts import FilePromptRepository


class PromptRepositoryTests(unittest.TestCase):
    def test_wolf_system_prompt_is_byte_for_byte_equivalent(self):
        bundle = FilePromptRepository().load("wolf_lumine", set())
        self.assertEqual(
            sha256(bundle.system_prompt.encode()).hexdigest(),
            "0aa3f3c54084ce6539234f9dbac9df833eae445fd585f873ee4dece5bfb4fb79",
        )

    def test_unknown_persona_falls_back_to_wolf(self):
        bundle = FilePromptRepository().load("unknown", set())
        self.assertEqual(bundle.persona_id, "wolf_lumine")
        self.assertEqual(
            sha256(bundle.system_prompt.encode()).hexdigest(),
            "2393ea4dff00f4dca046a2b09d07f983fac0ce2abba67d85b58fc84bdbd127de",
        )

    def test_archive_and_multimodal_snapshots_are_unchanged(self):
        repository = FilePromptRepository()
        archive = repository.load("存档读取", set()).system_prompt
        multimodal = repository.load("wolf_lumine", {"multimodal"}).system_prompt
        self.assertEqual(
            sha256(archive.encode()).hexdigest(),
            "a72c7b1c4d957282dfcbd14481af06eb453de3f325fdd2ae8fd0c91cc75c3737",
        )
        self.assertEqual(
            sha256(multimodal.encode()).hexdigest(),
            "7c2ea4082c110ae956c3e120a671f00bc1fddb214b5aacb73b0ae490fa6db6af",
        )
