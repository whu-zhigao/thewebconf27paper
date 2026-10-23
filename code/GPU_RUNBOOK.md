# GPU 云端全量实验手册（适配 CUDA 13.0 / Torch 2.7.0 / Python 3.11）

本包自包含：全部实验代码 + 7 个数据集 + 预计算关系索引 + 本手册。
目标：一次跑完论文中的**全部**实验，回传后自动更新论文所有表格。

## 0. 上传（工作站）

```bash
cd "/Volumes/workspace/VSworkspace/papers/Session Recommendation-WWW/www/code"
# 包已生成: mplignn_gpu_package.tar.gz (331MB)
scp mplignn_gpu_package.tar.gz user@gpu-server:~/
ssh user@gpu-server && tar xzf mplignn_gpu_package.tar.gz && cd code
```

## 1. 环境自检（9 项全 PASS 再继续）

```bash
python3 scripts/gpu_selftest.py
```

## 2. 一键全量实验（11 阶段，全部断点续跑）

```bash
nohup python3 run_all_gpu.py > gpu_all.log 2>&1 &
tail -f gpu_all.log
```

阶段明细（跳过已有结果，中断后重跑同一命令即可续）：

| 阶段 | 内容 | 规模 | 预计(GPU) |
|---|---|---|---|
| P1 | 预算协议主矩阵 | 14 模型 × 6 数据集 | 3–5 h |
| P2 | 五种子显著性 | 4 模型 × 6 数据集 × 5 种子 | 3–5 h |
| P3 | 消融（Diginetica） | 8 变体 | 1 h |
| P4 | **全轮协议**（无上限，30 epochs） | 6 模型 × 3 数据集 | 10–20 h |
| P5 | 敏感性扫描（L/h/d/α） | ~17 组 | 1–2 h |
| P6 | 噪声注入（0–50%） | 5 模型 | 0.5 h |
| P7 | 漂移分层评估 | MPLI+基线 | 0.5 h |
| P8 | GPU 效率基准 | 3 数据集 | 0.5 h |
| P9 | 审计/理论/案例/诊断/校准 | 6 项 | 1 h |
| P10 | 预研研究 P1/P2/P3 | 6 数据集 | 0.5 h |
| P11 | 打包 gpu_results.tar.gz | 全部产物 | — |

单独补跑某阶段：`python3 run_all_gpu.py --phases 4,6`

## 3. 回传（GPU 服务器）

日志出现 `ALL PHASES COMPLETE` 后：
```bash
scp gpu_results.tar.gz user@workstation:"/Volumes/workspace/VSworkspace/papers/Session Recommendation-WWW/www/code/"
```

## 4. 回写论文（工作站，一条命令）

```bash
cd "/Volumes/workspace/VSworkspace/papers/Session Recommendation-WWW/www/code"
python3 scripts/update_paper_results.py --tar gpu_results.tar.gz
```

自动完成：
- 解包校验全部结果 JSON
- **重写论文全部数据表**：主表(tab:main)、消融、噪声、附录 NDCG/Coverage、
  全轮协议表(full_protocol_table.tex)、五种子统计表(seed_statistics_table.tex)
- 生成 `results/real/PROSE_DIFF.md`：列出散文中引用的每一个数字是否变化，
  供人工同步摘要/引言/RQ 分析中的表述

```bash
cd .. && pdflatex main.tex && pdflatex main.tex   # 验证 0 errors / 0 '??'
python3 code/make_release.py                       # 重建可审计工件
```

## 故障排查

| 症状 | 处置 |
|---|---|
| 某组失败 | 看 `results/real/fail_*.txt`；修复后重跑总控自动续 |
| 中断/换机 | 直接重跑 `run_all_gpu.py`，已有 JSON 全部跳过 |
| 只想要全轮结果 | `run_all_gpu.py --phases 0,4,11`（最省时路径） |
| OOM | 保持 batch 128 不变（协议一致性优先） |
