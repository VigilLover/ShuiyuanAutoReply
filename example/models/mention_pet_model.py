import os
import json
import random
import logging
from typing import Optional

class MentionPetModel:
    """
    A model to handle 'rua' interactions (pet interactions) with persistent states.
    """

    def __init__(self, filepath="assets/pet_responses.json", state_path="assets/pet_state.json", endings_path="assets/pet_endings.json"):
        self.filepath = filepath
        self.state_path = state_path
        self.endings_path = endings_path

    def _load_state(self) -> dict:
        """加载宠物持久化状态。如果不存在则初始化。"""
        default_state = {"patience": 0, "wisdom": 0, "chaos": 0}
        if not os.path.exists(self.state_path):
            return default_state
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"==> [MentionPetModel] State load error: {str(e)}")
            return default_state

    def _save_state(self, state: dict):
        """保存状态到本地文件。"""
        try:
            os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"==> [MentionPetModel] State save error: {str(e)}")

    def _clamp_stat(self, value: int, min_val: int = -100, max_val: int = 100) -> int:
        """限制属性值在 -100 到 100 之间"""
        return max(min_val, min(max_val, value))

    def _generate_progress_bar(self, label: str, value: int, delta: int) -> str:
        """生成带具体数值和增减提示的等长进度条"""
        length = 20
        # 将 -100 ~ 100 映射为 0 ~ length 的进度
        filled = int((value + 100) / 200 * length)
        # 防止越界
        filled = max(0, min(length, filled))
        bar = "■" * filled + "□" * (length - filled)
        
        delta_str = f"+{delta}" if delta >= 0 else f"{delta}"
        return f"{label} [{bar}] {value:>4} ({delta_str})"

    def get_rua_response(self, username: str = None, name: str = None) -> Optional[str]:
        """
        读取状态与配置文件，生成带 ASCII 表情和变动结果格式化文本。
        特定用户触发可以获得不同的文案和特定属性值增减。
        """
        try:
            if not os.path.exists(self.filepath):
                logging.error(f"==> [MentionPetModel] File not found: {self.filepath}")
                return "（找不到心情配置文件哦...）"

            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not data:
                return "（目前没有任何心情呢...）"

            # 准备概率抽签
            states = list(data.keys())
            weights = [info.get("weight", 10) for info in data.values()]
            
            # 使用权重选择当前心情
            selected_state = random.choices(states, weights=weights, k=1)[0]
            action_data = data[selected_state]
            
            # 检查是否有对当前特定用户的覆盖逻辑 (special_overrides)
            overrides = action_data.get("special_overrides", {})
            user_override = None
            if name and name in overrides:
                user_override = overrides[name]
            elif username and username in overrides:
                user_override = overrides[username]

            if user_override:
                texts = user_override.get("texts", action_data.get("texts", []))
                asciis = user_override.get("ascii", action_data.get("ascii", [""]))
                deltas = user_override.get("deltas", action_data.get("deltas", {}))
            else:
                texts = action_data.get("texts", [])
                asciis = action_data.get("ascii", [""])
                deltas = action_data.get("deltas", {})
            
            selected_text = random.choice(texts) if texts else "（发呆中）"
            selected_ascii = random.choice(asciis) if asciis else ""

            # 加载并变更状态
            state = self._load_state()
            
            patience_delta = deltas.get("patience", 0)
            wisdom_delta = deltas.get("wisdom", 0)
            chaos_delta = deltas.get("chaos", 0)

            # 更新并截断属性值
            state["patience"] = self._clamp_stat(state.get("patience", 0) + patience_delta)
            state["wisdom"]   = self._clamp_stat(state.get("wisdom",   0) + wisdom_delta)
            state["chaos"]    = self._clamp_stat(state.get("chaos",    0) + chaos_delta)

            # ===== 隐藏结局判定逻辑 =====
            limits_hit = []
            for stat in ["patience", "wisdom", "chaos"]:
                if state[stat] >= 100:
                    limits_hit.append(f"{stat}_max")
                elif state[stat] <= -100:
                    limits_hit.append(f"{stat}_min")
                    
            if limits_hit:
                # 判定具体是单一触发还是多重触发彩蛋
                ending_id = "multiple" if len(limits_hit) >= 2 else limits_hit[0]
                
                # 获取该结局信息
                endings_data = {}
                if os.path.exists(self.endings_path):
                    with open(self.endings_path, "r", encoding="utf-8") as fe:
                        endings_data = json.load(fe)
                        
                ending_info = endings_data.get(ending_id)
                if ending_info:
                    # 触发结局后，将所有属性清零并保存
                    self._save_state({"patience": 0, "wisdom": 0, "chaos": 0})
                    
                    response_lines = [
                        "```text",
                        ending_info.get("ascii", ""),
                        "",
                        f"{ending_info.get('title', '【结局】')} {ending_info.get('text', '')}",
                        "",
                        "（已达成以上结局，所有属性重新归零）",
                        "```"
                    ]
                    return "\n".join([line for line in response_lines if line is not None])

            # 如果没有到达极限值，正常进行保存及回复构造
            self._save_state(state)

            # 构造最终 Markdown 回复块 (包含 ```)
            response_lines = [
                "```text",
                selected_ascii,
                "",
                f"【{selected_state}】 {selected_text}",
                "",
                self._generate_progress_bar("耐心", state['patience'], patience_delta),
                self._generate_progress_bar("智慧", state['wisdom'], wisdom_delta),
                self._generate_progress_bar("混沌", state['chaos'], chaos_delta),
                "```"
            ]

            return "\n".join([line for line in response_lines if line is not None])
            
        except Exception as e:
            logging.error(f"==> [MentionPetModel] Error processing rua response: {str(e)}")
            return "（被rua了一下，但是似乎出了点小错误）"