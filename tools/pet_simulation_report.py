"""
使用方式：

1) 基本运行（使用默认配置路径与参数）
    python tools/pet_simulation_report.py

2) 指定模拟次数与随机种子
    python tools/pet_simulation_report.py --trials 20000 --seed 42

3) 指定每局最大步数
    python tools/pet_simulation_report.py --max-steps 1500

4) 导出结构化 JSON 报告
    python tools/pet_simulation_report.py --json-out reports/pet_report.json

5) 使用自定义配置文件
    python tools/pet_simulation_report.py --responses assets/pet_responses.json --endings assets/pet_endings.json
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


def apply_random_offset(value: int, rng: random.Random) -> int:
    if value == 0:
        return 0
    offset = rng.randint(-5, 5)
    new_value = value + offset
    if value > 0:
        return max(1, new_value)
    return min(-1, new_value)


def clamp_stat(value: int, minimum: int = -100, maximum: int = 100) -> int:
    return max(minimum, min(maximum, value))


def load_json(file_path: Path) -> dict:
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def infer_stat_names(responses: dict, endings: dict) -> List[str]:
    stat_names = set()

    for state_info in responses.values():
        for stat_name in state_info.get("deltas", {}).keys():
            stat_names.add(stat_name)

    for ending_id in endings.keys():
        if ending_id.endswith("_max"):
            stat_names.add(ending_id[: -len("_max")])
        elif ending_id.endswith("_min"):
            stat_names.add(ending_id[: -len("_min")])

    if not stat_names:
        return ["patience", "wisdom", "chaos"]
    return sorted(stat_names)


def choose_ending_id(state: Dict[str, int], stat_names: List[str], endings: dict) -> str:
    limits_hit = []
    for stat_name in stat_names:
        value = state.get(stat_name, 0)
        if value >= 100:
            limits_hit.append(f"{stat_name}_max")
        elif value <= -100:
            limits_hit.append(f"{stat_name}_min")

    if not limits_hit:
        return ""

    ending_id = "multiple" if len(limits_hit) >= 2 else limits_hit[0]
    return ending_id if ending_id in endings else ""


def simulate(
    responses: dict,
    endings: dict,
    trials: int,
    max_steps: int,
    seed: int | None,
) -> Tuple[Counter, Counter, int, float]:
    rng = random.Random(seed)
    state_names = list(responses.keys())
    state_weights = [responses[name].get("weight", 10) for name in state_names]
    stat_names = infer_stat_names(responses, endings)

    mood_trigger_counter: Counter = Counter()
    ending_counter: Counter = Counter()
    unfinished = 0
    total_steps_until_end = 0

    for _ in range(trials):
        stats = {name: 0 for name in stat_names}
        finished = False

        for step in range(1, max_steps + 1):
            selected_state = rng.choices(state_names, weights=state_weights, k=1)[0]
            mood_trigger_counter[selected_state] += 1

            deltas = responses[selected_state].get("deltas", {})
            for stat_name in stat_names:
                current_value = stats.get(stat_name, 0)
                base_delta = deltas.get(stat_name, 0)
                delta_with_offset = apply_random_offset(base_delta, rng)
                stats[stat_name] = clamp_stat(current_value + delta_with_offset)

            ending_id = choose_ending_id(stats, stat_names, endings)
            if ending_id:
                ending_counter[ending_id] += 1
                total_steps_until_end += step
                finished = True
                break

        if not finished:
            unfinished += 1

    finished_trials = trials - unfinished
    average_steps = (total_steps_until_end / finished_trials) if finished_trials > 0 else 0.0
    return mood_trigger_counter, ending_counter, unfinished, average_steps


def format_report(
    responses: dict,
    endings: dict,
    trials: int,
    max_steps: int,
    seed: int | None,
    mood_trigger_counter: Counter,
    ending_counter: Counter,
    unfinished: int,
    average_steps: float,
) -> str:
    lines = []
    lines.append("=== 宠物心情模拟报告 ===")
    lines.append(f"试验次数: {trials}")
    lines.append(f"单局最大步数: {max_steps}")
    lines.append(f"随机种子: {seed if seed is not None else 'None'}")
    lines.append("")

    total_mood_triggers = sum(mood_trigger_counter.values())
    lines.append("--- 心情状态触发次数 ---")
    for state_name in responses.keys():
        count = mood_trigger_counter.get(state_name, 0)
        ratio = (count / total_mood_triggers * 100) if total_mood_triggers > 0 else 0.0
        lines.append(f"{state_name}: {count} ({ratio:.2f}%)")
    lines.append("")

    finished_trials = trials - unfinished
    lines.append("--- 最终结局概率报告 ---")
    for ending_id in endings.keys():
        count = ending_counter.get(ending_id, 0)
        prob_by_all = count / trials * 100 if trials > 0 else 0.0
        prob_by_finished = count / finished_trials * 100 if finished_trials > 0 else 0.0
        lines.append(
            f"{ending_id}: {count} | 占全部试验 {prob_by_all:.2f}% | 占已结束试验 {prob_by_finished:.2f}%"
        )

    if unfinished > 0:
        unfinished_ratio = unfinished / trials * 100 if trials > 0 else 0.0
        lines.append(f"未在步数上限内结束: {unfinished} ({unfinished_ratio:.2f}%)")

    lines.append(f"平均触发步数(仅统计已结束): {average_steps:.2f}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="模拟宠物心情状态并输出结局概率报告")
    parser.add_argument(
        "--responses",
        type=Path,
        default=repo_root / "assets" / "pet_responses.json",
        help="心情配置文件路径",
    )
    parser.add_argument(
        "--endings",
        type=Path,
        default=repo_root / "assets" / "pet_endings.json",
        help="结局配置文件路径",
    )
    parser.add_argument("--trials", type=int, default=10000, help="模拟局数")
    parser.add_argument("--max-steps", type=int, default=1000, help="每局最大步数")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="可选，输出结构化 JSON 报告文件路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.trials <= 0:
        raise ValueError("--trials 必须大于 0")
    if args.max_steps <= 0:
        raise ValueError("--max-steps 必须大于 0")

    responses = load_json(args.responses)
    endings = load_json(args.endings)

    mood_trigger_counter, ending_counter, unfinished, average_steps = simulate(
        responses=responses,
        endings=endings,
        trials=args.trials,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    report_text = format_report(
        responses=responses,
        endings=endings,
        trials=args.trials,
        max_steps=args.max_steps,
        seed=args.seed,
        mood_trigger_counter=mood_trigger_counter,
        ending_counter=ending_counter,
        unfinished=unfinished,
        average_steps=average_steps,
    )
    print(report_text)

    if args.json_out is not None:
        finished_trials = args.trials - unfinished
        payload = {
            "trials": args.trials,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "mood_trigger_counts": dict(mood_trigger_counter),
            "ending_counts": dict(ending_counter),
            "ending_probability_all_trials": {
                ending_id: (ending_counter.get(ending_id, 0) / args.trials)
                for ending_id in endings.keys()
            },
            "ending_probability_finished_trials": {
                ending_id: (ending_counter.get(ending_id, 0) / finished_trials)
                if finished_trials > 0
                else 0.0
                for ending_id in endings.keys()
            },
            "unfinished": unfinished,
            "average_steps_finished_trials": average_steps,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
