# CODEX_CONTEXT_P1_C

- P1-C Goal 自动验收：PASS。
- H-C1 r2=24/24；H-C2 r3=36/36。
- 正式数据：output_generation_p1_c_clean_context_7200_strict_concept_r2，7200 样本，旧几何复用 0。
- Stage6：6/6 三种子正式运行通过；preholdout selection SHA-256=25de1a6198f4defc5dc865871685692c82dfb4121793a73e26b382d762cfbcc0。
- Real-Dev F1=0.446215，低于 P1-B 0.600；rib recall=0；不得据此调参。
- Holdout-v3：正式推理 1 次；P1-C object Macro-F1=0.941987，相对 P1-B Δ=+0.273036。
- 受保护 P1-B 资产逐文件不变；P1-C tests 68/68 PASS。
- 导入顺序复发已由 p1_c/runtime_state_isolation_v3.py 和 tests/test_p1_c_import_order_isolation.py 封堵。
- 禁止项全部未使用；P2 未进入。
