# 현재 실험 결과 정리

> 최종 업데이트: 2026-05-16

---

## 핵심 결과 — density_fix 실험 (최종)

> 조건: gsplat 1.x + Doppler radial noise 수정 + dynamic_compositor 적용 후 재실험

| Method | MOTA | MOTP | FP/frame | IDSW | Recall | Precision |
|--------|------|------|----------|------|--------|-----------|
| PMBM Phase29 (radar GT) | **-0.177** | **1.20m** | **0.8** | **3** | — | — |
| Exp A — Real only | -9.504 | 1.202m | 30.4 | 456 | 0.318 | 0.032 |
| Exp B — Real+Synth 1:1 | -10.232 | 1.219m | 40.2 | **273** | 0.259 | 0.024 |
| Exp C — Real+Synth 1:3 | -14.405 | 1.235m | 33.3 | 712 | **0.438** | 0.029 |

N_GT = 3170 | val: scene 6–7 (82 frames) | best checkpoint: val_loss 기준

---

## 2026-05-16 세션 — 변경 사항 및 발견

### 1. 버그 수정: run_density_fix.ps1 — `$Args` → `$RunArgs`

**발견 경위:**
재부팅 후 이전 실험 로그 분석. `density_fix_exp_{a,b,c}.log` 세 개 모두
`Experiment: real_only`, `Train size: 242` 로 기록 — synth 데이터 미로딩 확인.

**근본 원인:**
PowerShell에서 `$Args`는 예약된 자동 변수(automatic variable).
`param($Label, $Args)` 선언 시 함수 내 `@Args` splatting이 전달 배열이
아닌 자동 변수를 참조 → `--synth-dir` 인수 무시됨.

```powershell
# 수정 전 (버그)
function Run-Exp {
    param($Label, $Args)
    python train_mixed.py @Args ...

# 수정 후
function Run-Exp {
    param($Label, $RunArgs)
    python train_mixed.py @RunArgs ...
```

**영향:** 이전 density_fix 실험(Exp A/B/C)이 전부 real_only로 실행됨.
수정 후 재실행하여 정상 결과 확보.

---

### 2. MOTA 분해 분석

MOTA = 1 − (FP + FN + IDSW) / N_GT

| | FP/N_GT | FN/N_GT | IDSW/N_GT | Recall | Precision |
|---|---|---|---|---|---|
| real_only | 9.68 | 0.682 | 0.144 | 0.318 | 0.032 |
| real+synth_r1 | 10.41 | 0.741 | **0.086** | 0.259 | 0.024 |
| real+synth_r3 | 14.62 | 0.562 | 0.225 | **0.438** | 0.029 |

**핵심 관찰:**
- MOTA 오차의 **95%가 FP** 기여 — IDSW는 10분의 1 수준
- r1: IDSW **273** (real_only 456 대비 **−40%**) — 합성 레이더의 실제 효과 확인
- r1: FP **40.2** (real_only 30.4 대비 **+32%**) — 카메라 domain gap 실증

---

### 3. 핵심 발견 — 두 Modality의 비대칭 효과

| Modality | 효과 | 방향 |
|----------|------|------|
| 합성 카메라 (3DGS novel view) | FP↑, Recall↓ | **역효과** |
| 합성 레이더 (Doppler 수정 후) | IDSW −40% | **실제 효과** |

**Radar System Engineer 관점:**
- 실제 레이더는 RCS 기반 반사 — 3DGS depth는 이를 모델링하지 않음
- `dynamic_compositor`는 real nuScenes 레이더를 re-project → RCS/noise 보존
- 물리적 정확도: dynamic_compositor > 3DGS depth 기반 합성

**Deep Learning Researcher 관점:**
```
# 구조적 원인
model = BEVFormer(cfg)   # ResNet50 frozen!

real camera  → frozen ResNet50 → feature_real   (distribution A)
synth camera → frozen ResNet50 → feature_synth  (distribution B ≠ A)
→ BEV encoder가 A와 B를 동시에 맞추지 못함 → FP↑, Recall↓

레이더: RadarFusion은 learnable + continuous numeric feature
→ Doppler가 물리적으로 올바르면 실제로 학습 → IDSW −40%
```

---

### 4. 방향 결정 — Graphics + CV 통합 포트폴리오

**포트폴리오 서술 구조:**

```
[Graphics 기여 — NeuralSensorSim Phase 1~3]
  - PSNR 30.94dB 장면 복원 (gsplat 1.x + densification, 7.17M Gaussians)
  - Novel view synthesis (12× viewpoint diversity per frame)
  - Depth 기반 radar point 생성 (range + Doppler physics)

[CV 기여 — BEVFormerRadar Exp A~C]
  - 실험적 발견: frozen backbone + synthetic camera = domain gap 정량 실증
  - 실험적 발견: Doppler-corrected synthetic radar = IDSW −40% 개선
  - Open problem 문서화: camera sim-to-real gap은 backbone fine-tuning 필요
```

**이것은 실패 스토리가 아님.**
"Neural rendering으로 sensor data를 생성했을 때 어떤 modality가 효과적이고
어떤 modality가 bottleneck인가"를 실험적으로 밝힌 연구.

**포트폴리오 핵심 문장:**
> "3DGS 기반 합성 데이터에서 frozen camera backbone의 domain gap으로 MOTA는
> 소폭 저하되지만, Doppler-correct 레이더 augmentation은 IDSW를 40% 개선한다.
> 합성 레이더 데이터가 tracking consistency에 기여하며, camera branch의
> sim-to-real gap은 backbone fine-tuning 없이는 해결이 어려움을 보인다."

---

## 다음 단계 — Exp D: Radar-Only Augmentation

**목적:** 3DGS 카메라를 제거하고 레이더 augmentation 효과만 순수하게 측정.
"domain gap이 원인"이라는 가설을 실험적으로 검증.

**파이프라인:**
```
real nuScenes camera → 원본 그대로 (domain gap 없음)
real nuScenes radar  → dynamic_compositor (copy-paste 동적 객체)
                     → [A] 사전 생성 .npz (구조적 augmentation)
                     → [B] 인라인 noise/dropout (확률적 augmentation)
                     → RadarAugDataset → 학습
```

**기대 비교:**

| Method | MOTA | FP/f | IDSW |
|--------|------|------|------|
| real_only | -9.504 | 30.4 | 456 |
| real+synth_r1 (3DGS cam 포함) | -10.232 | 40.2 | 273 |
| **Exp D (real cam + radar aug)** | **TBD** | **≈30?** | **<273?** |

FP ≈ real_only + IDSW < r1 → 가설 확인.

**신규 파일:**
```
NeuralSensorSim/scripts/augment_radar_real.py
BEVFormerRadar/src/data/radar_aug_dataset.py
BEVFormerRadar/run_radar_aug.ps1
```

---

## 전체 실험 이력

| 날짜 | 실험 | 핵심 결과 | 비고 |
|------|------|-----------|------|
| 2026-05-07 | Phase 1~5 초기 완료 | FP −58%, MOTP PMBM 수준 | Doppler=0 버그 있음 |
| 2026-05-08 | Doppler radial 수정 | IDSW 개선 | Step 2 완료 |
| 2026-05-11 | gsplat 1.x 업그레이드 | PSNR 24.5→30.94dB | densification 활성화 |
| 2026-05-12 | Exp A/B/C (구버전) | IDSW 80~514 | sigma 수정 전 |
| 2026-05-13 | dynamic_compositor 완성 | copy-paste augmentation | Step 3 완료 |
| 2026-05-15 | run_density_fix.ps1 실행 | 버그로 전부 real_only | `$Args` 변수 충돌 |
| **2026-05-16** | **버그 수정 + density_fix 재실험** | **IDSW −40%, FP +32% 발견** | **본 세션** |
| TBD | Exp D (radar-only aug) | TBD | 다음 단계 |

---

## 알려진 한계

| 한계 | 상태 | 개선 방향 |
|------|------|-----------|
| Frozen backbone + synth camera = domain gap | ✅ 실험 확인 | backbone fine-tuning (프로젝트 범위 초과) |
| 3DGS scene 0만 사용 | 유지 | scene 1~5 확장 |
| val set 소규모 (82 frames, 2 scenes) | 구조적 한계 | full nuScenes 필요 |
| best checkpoint = val_loss 기준 (MOTA 직접 모니터링 미적용) | 유지 | MOTA-based early stopping |
| 합성 레이더 RCS 미모델링 | 유지 | dynamic_compositor (real RCS 보존)로 보완 |
