# 实验 3：P-LOPO 留一题 24 折

本实验没有独立脚本，其数据由主入口脚本生成：

- 脚本：`code/02_exp1_non_inferiority/run_block1.py` 中的 `protocol_lopo()`
- 内容：ASAP-1 上留 1 题作测试、其余 7 题训练，8 题 × 3 种子 = 24 折，
  逐折输出 BlackBox / H2T-CBD 的 QWK 与配对差值 ΔQWK。
- 结果：保存在 `results/runs/B1_LOPO_feat_{blackbox,h2tcbd}_s{seed}_fold{fold}/result.json`
  以及整合表 `data/03_experiment_results_integration.xlsx` 的「Exp3_LOPO 24折」sheet。

运行方式（在打包目录根下）：

```bash
cd code/02_exp1_non_inferiority
python run_block1.py   # 会按需跳过已完成单元，包含 P-in / P-LOPO / P-cross
```

注意：嵌入缓存已随附（`results/runs/_emb/*.npz`）；如需重新生成，用
`code/00_core/extract_embeddings.py`（需要 GPU）。
