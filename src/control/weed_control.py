# 除草策略联动：根据细粒度分类结果（杂草种类、根深、生长阶段）输出功率/时间/深度

import os
from typing import Dict, Any, Optional

# 默认策略表：杂草类型 -> [深根, 中根, 浅根] 每项 (功率W, 时间ms, 深度cm)
DEFAULT_STRATEGIES = {
    0: [(250, 80, 10), (200, 70, 8), (180, 60, 6)],   # 稗草
    1: [(150, 50, 5), (130, 40, 4), (120, 30, 3)],   # 马齿苋
    2: [(200, 60, 8), (180, 50, 6), (160, 40, 5)],   # 马唐
    3: [(180, 55, 7), (160, 45, 6), (140, 35, 4)],
    4: [(220, 70, 9), (190, 60, 7), (170, 50, 5)],
    5: [(160, 48, 6), (140, 38, 5), (130, 32, 3)],
    6: [(200, 65, 8), (175, 55, 6), (155, 45, 5)],
    7: [(170, 52, 6), (150, 42, 5), (135, 35, 4)],
}


def weed_control_strategy(
    weed_id: int,
    depth_level: int,
    growth_stage: int,
    strategies: Optional[Dict[int, list]] = None,
    seedling_power_scale: float = 0.8,
) -> Dict[str, Any]:
    """
    根据细粒度分类结果生成除草执行参数。
    :param weed_id: 杂草类别 0~7
    :param depth_level: 根深等级 0=深 1=中 2=浅
    :param growth_stage: 作物生长阶段 0=幼苗 1=成株
    :param strategies: 策略表，缺省用 DEFAULT_STRATEGIES
    :param seedling_power_scale: 幼苗期功率/时间缩放
    :return: {"power": W, "duration": ms, "depth": cm}
    """
    strategies = strategies or DEFAULT_STRATEGIES
    depth_level = max(0, min(2, depth_level))
    weed_id = weed_id % len(strategies) if strategies else 0
    row = strategies.get(weed_id, DEFAULT_STRATEGIES[0])
    power, duration, depth = row[depth_level]

    if growth_stage == 0:
        power = int(power * seedling_power_scale)
        duration = int(duration * seedling_power_scale)

    return {"power": power, "duration": duration, "depth": depth}


def load_weed_control_config(config_path: Optional[str] = None) -> tuple:
    """从 YAML 加载 weed_control 配置，返回 (strategies_dict, seedling_scale)."""
    strategies = dict(DEFAULT_STRATEGIES)
    seedling_scale = 0.8
    if config_path is None:
        try:
            config_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "config", "default.yaml"
            )
        except Exception:
            return strategies, seedling_scale
    if not os.path.isfile(config_path):
        return strategies, seedling_scale
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        weed_cfg = cfg.get("weed_control") or {}
        for k, v in weed_cfg.items():
            if k == "seedling_power_scale":
                seedling_scale = float(v)
            elif isinstance(v, list) and len(v) > 0:
                try:
                    ki = int(k)
                    strategies[ki] = v
                except (TypeError, ValueError):
                    pass
        seedling_scale = float(weed_cfg.get("seedling_power_scale", seedling_scale))
    except Exception:
        pass
    return strategies, seedling_scale
