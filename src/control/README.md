# control — 除草策略与控制

根据检测模块的**细粒度输出**（杂草种类、根深等级、作物生长阶段）生成执行参数：激光功率、作用时间、切割深度等，并与配置中的策略表联动。

---

## 文件说明

| 文件 | 说明 |
|------|------|
| **weed_control.py** | 除草策略实现：默认策略表（按杂草类型与根深等级）、`weed_control_strategy()` 生成单次执行参数、`load_weed_control_config()` 从 YAML 加载配置。 |

---

## 主要接口

- **weed_control_strategy(weed_id, depth_level, growth_stage, strategies=None, seedling_power_scale=0.8)**  
  返回 `{"power": W, "duration": ms, "depth": cm}`。

- **load_weed_control_config(config_path=None)**  
  返回 `(strategies_dict, seedling_scale)`，策略表可与 `weed_control_strategy` 的 `strategies` 参数配合使用。

配置格式见项目根目录下 `config/default.yaml` 中的 `weed_control` 段。
