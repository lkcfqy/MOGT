# Synthetic Multi-Slot Standard Summary

Date: 2026-05-04

This summary is generated only from benchmark artifacts that contain the
`mogt-experiment-v1` standard report schema.

## mamba_d192_slots4_final_curriculum

Model: `mamba`. Seeds: 7, 42, 123. Steps: 3000. Train context: 512. Slots: 4. Dense loss: False.

| Context | Accuracy | Seeds |
|---:|---:|---:|
| 512 | 17.71% +/- 10.97% | 3 |
| 1024 | 20.83% +/- 9.92% | 3 |
| 2048 | 15.62% +/- 5.63% | 3 |
| 4096 | 19.79% +/- 10.97% | 3 |

Artifacts:
- `benchmark_runs/synthetic_multislot4_mamba_d192_slotcurr_finalonly_stdreport_ctx512_seed123_steps3000.json`
- `benchmark_runs/synthetic_multislot4_mamba_d192_slotcurr_finalonly_stdreport_ctx512_seed42_steps3000.json`
- `benchmark_runs/synthetic_multislot4_mamba_d192_slotcurr_finalonly_stdreport_ctx512_seed7_steps3000.json`

## mogt_identity_coupled_write_forget_current_prev_prefix_value_bias_m2p0_slots4_final

Model: `mogt`. Seeds: 7, 42, 123. Steps: 3000. Train context: 512. Slots: 4. Dense loss: False.

| Context | Accuracy | Seeds |
|---:|---:|---:|
| 512 | 44.27% +/- 14.52% | 3 |
| 1024 | 38.54% +/- 21.89% | 3 |
| 2048 | 41.67% +/- 22.61% | 3 |
| 4096 | 44.79% +/- 17.21% | 3 |

Artifacts:
- `benchmark_runs/synthetic_multislot4_mogt_slotaddr_coupled_rank_finalonly_stdreport_ctx512_seed123_steps3000.json`
- `benchmark_runs/synthetic_multislot4_mogt_slotaddr_coupled_rank_finalonly_stdreport_ctx512_seed42_steps3000.json`
- `benchmark_runs/synthetic_multislot4_mogt_slotaddr_coupled_rank_finalonly_stdreport_ctx512_seed7_steps3000.json`

## mogt_identity_coupled_write_forget_current_prev_prefix_value_bias_m2p0_slots4_final_curriculum

Model: `mogt`. Seeds: 7, 42, 123. Steps: 3000. Train context: 512. Slots: 4. Dense loss: False.

| Context | Accuracy | Seeds |
|---:|---:|---:|
| 512 | 100.00% +/- 0.00% | 3 |
| 1024 | 100.00% +/- 0.00% | 3 |
| 2048 | 99.48% +/- 0.90% | 3 |
| 4096 | 94.27% +/- 2.39% | 3 |

Artifacts:
- `benchmark_runs/synthetic_multislot4_mogt_slotaddr_coupled_rank_slotcurr_finalonly_stdreport_ctx512_seed123_steps3000.json`
- `benchmark_runs/synthetic_multislot4_mogt_slotaddr_coupled_rank_slotcurr_finalonly_stdreport_ctx512_seed42_steps3000.json`
- `benchmark_runs/synthetic_multislot4_mogt_slotaddr_coupled_rank_slotcurr_finalonly_stdreport_ctx512_seed7_steps3000.json`

## transformer_nope_slots4_final

Model: `transformer`. Seeds: 7, 42, 123. Steps: 3000. Train context: 512. Slots: 4. Dense loss: False.

| Context | Accuracy | Seeds |
|---:|---:|---:|
| 512 | 26.56% +/- 2.71% | 3 |
| 1024 | 24.48% +/- 3.93% | 3 |
| 2048 | 21.35% +/- 3.25% | 3 |
| 4096 | 25.00% +/- 5.63% | 3 |

Artifacts:
- `benchmark_runs/synthetic_multislot4_transformer_nope_finalonly_stdreport_ctx512_seed123_steps3000.json`
- `benchmark_runs/synthetic_multislot4_transformer_nope_finalonly_stdreport_ctx512_seed42_steps3000.json`
- `benchmark_runs/synthetic_multislot4_transformer_nope_finalonly_stdreport_ctx512_seed7_steps3000.json`

## transformer_nope_slots4_final_curriculum

Model: `transformer`. Seeds: 7, 42, 123. Steps: 3000. Train context: 512. Slots: 4. Dense loss: False.

| Context | Accuracy | Seeds |
|---:|---:|---:|
| 512 | 31.25% +/- 5.41% | 3 |
| 1024 | 29.69% +/- 5.63% | 3 |
| 2048 | 23.44% +/- 5.63% | 3 |
| 4096 | 21.35% +/- 6.31% | 3 |

Artifacts:
- `benchmark_runs/synthetic_multislot4_nope_slotcurr_finalonly_stdreport_ctx512_seed123_steps3000.json`
- `benchmark_runs/synthetic_multislot4_nope_slotcurr_finalonly_stdreport_ctx512_seed42_steps3000.json`
- `benchmark_runs/synthetic_multislot4_nope_slotcurr_finalonly_stdreport_ctx512_seed7_steps3000.json`
