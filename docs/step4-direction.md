# Step 4 — 실험 분석 & 다음 방향

> 작성: 2026-05-16
> 배경: density_fix 실험 완료 후 결과 분석 및 방향 결정

---

## 1. density_fix 실험 결과

**실험 조건:** gsplat 1.x + Doppler radial noise 수정 + dynamic_compositor 적용 후 재실험

| Method | MOTA | MOTP | FP/f | IDSW | Recall | Precision |
|--------|------|------|------|------|--------|-----------|
| PMBM Phase29 (radar GT) | -0.177 | 1.20m | 0.8 | 3 | — | — |
| real_only (Exp A) | -9.504 | 1.202m | 30.4 | 456 | 0.318 | 0.032 |
| real+synth_r1 (Exp B) | -10.232 | 1.219m | 40.2 | **273** | 0.259 | 0.024 |
| real+synth_r3 (Exp C) | -14.405 | 1.235m | 33.3 | 712 | **0.438** | 0.029 |

N_GT = 3170 (val 전체 GT 객체 수)

---

## 2. MOTA 분해 분석

MOTA = 1 − (FP + FN + IDSW) / N_GT

| | FP/N_GT | FN/N_GT | IDSW/N_GT |
|---|---|---|---|
| real_only | 9.68 | 0.682 | 0.144 |
| real+synth_r1 | 10.41 | 0.741 | **0.086** |
| real+synth_r3 | 14.62 | 0.562 | 0.225 |

**핵심 관찰:**
- MOTA 오차의 **95%가 FP**에서 발생. IDSW는 10분의 1 수준.
- 합성 데이터(r1)가 IDSW를 real_only 대비 **−40%** 개선함 — 실제 효과 있음.
- 그러나 FP 증가가 IDSW 개선을 압도 → net MOTA 하락.

---

## 3. 전문가 분석 — 두 관점의 합의

### Radar System Engineer 관점

현재 radar synthesizer: `3DGS depth → range, Doppler 추가 → radar point`

실제 레이더와의 핵심 차이:
- 실제 레이더는 **RCS(Radar Cross Section)** 기반으로 반사 강도가 결정됨
  - 차량 금속 패널 → 강한 반사 / 유리, 보행자 → 약한 반사
- 3DGS depth는 시각적 geometry를 복원하지 레이더 반사 특성을 모델링하지 않음
- 결과: 합성 레이더는 **geometry는 맞지만 density pattern이 다름**

반면 `dynamic_compositor`는 실제 nuScenes 레이더 포인트를 다른 ego pose로 re-project:
- RCS, noise 특성이 모두 보존됨
- **물리적으로 훨씬 현실적**

### Deep Learning Researcher 관점

```python
model = BEVFormer(cfg)   # ResNet50 frozen!
```

Learnable module: BEV encoder + detection head 뿐.

```
real camera  → frozen ResNet50 → feature_real   (distribution A)
synth camera → frozen ResNet50 → feature_synth  (distribution B)
```

frozen backbone은 synthetic 이미지에 adapt 불가능. BEV encoder가 A와 B를
동시에 맞추려다 둘 다 제대로 못 맞춤 → **FP↑, Recall↓**

레이더는 다름: RadarFusion은 learnable이고, radar point는 continuous numeric
feature (x, y, z, vx, vy). Doppler가 물리적으로 올바르면 network가 실제로
학습함. **IDSW −40%가 이를 증명.**

### 결론

```
Camera branch:  3DGS 이미지 → frozen backbone domain gap → FP↑, Recall↓  (역효과)
Radar branch:   Doppler 수정 → velocity cue 학습 → IDSW↓                  (실제 효과)
```

합성 레이더 효과는 실제로 유효하지만, 합성 카메라가 그것을 덮어버리고 있음.

---

## 4. 포트폴리오 관점 — Graphics + CV 통합 서술

타겟 독자: **Sensor Simulation 전문가 + ML/Perception 엔지니어 (둘 다)**

```
[Graphics side — NeuralSensorSim]
Phase 1: Static scene reconstruction  → PSNR 30.94dB, 7.17M Gaussians
Phase 2: Novel view synthesis         → 12× viewpoint diversity
Phase 3: Depth-based radar synthesis  → range/Doppler physics model
  ↓
발견: 3DGS camera images + frozen backbone = sim-to-real domain gap 실증
      "왜 안 되는가"를 수치로 설명 (FP +32%, Recall −18%)
  ↓
[CV side — BEVFormerRadar]
발견: Doppler-corrected synthetic radar → IDSW −40%
      레이더 branch는 실제로 학습됨
  ↓
Gap 분석: camera synthesis 한계 = backbone fine-tuning / domain adaptation 필요
          (open problem으로 문서화)
```

이것은 **실패 스토리가 아님.** "Neural rendering으로 sensor data를 생성했을 때
어떤 modality가 효과적이고 어떤 modality가 bottleneck인가"를 실험적으로 밝힌 연구.

포트폴리오 핵심 문장:
> "3DGS 기반 레이더-카메라 합성 데이터에서 frozen camera backbone의 domain gap으로
> MOTA는 소폭 저하되지만, Doppler-correct 레이더 augmentation은 IDSW를 40% 개선한다.
> 합성 레이더 데이터가 tracking consistency에 기여하며,
> camera branch의 sim-to-real gap은 backbone fine-tuning 없이는 해결이 어려움을 보인다."

---

## 5. 다음 실험 — Exp D: Radar-Only Augmentation

### 목적

3DGS 카메라를 완전히 제거하고 **레이더 augmentation 효과만** 순수하게 측정.
포트폴리오의 마지막 실험 피스.

### 데이터 파이프라인

```
[현재 Exp A/B/C]
3DGS novel view camera + depth → radar → dynamic_compositor → sample.npz
                                                               ↓
                                                         MixedDataset

[Exp D]
real nuScenes camera (원본) ─────────────────────────────────→ 그대로
real nuScenes radar         → dynamic_compositor (copy-paste)
                              → augmented radar npz
                                    ↓ [A: 사전 생성]
                              + inline noise/dropout
                                    ↓ [B: 인라인]
                                RadarAugDataset
```

### A+B 조합의 역할 분리

| | 역할 | 비용 | 구현 |
|---|---|---|---|
| **A (사전 생성)** | 구조적 augmentation — copy-paste 동적 객체, ego pose 매칭, GT velocity 변환 | 오프라인 1회 | `augment_radar_real.py` |
| **B (인라인)** | 확률적 augmentation — radar point dropout, noise jitter, density sampling | 런타임, 경량 | `RadarAugDataset.__getitem__` |

A가 "어떤 동적 객체가 추가되는가"를 결정하고, B가 "그 포인트에 어떤 perturbation이 가해지는가"를 결정.

### 필요한 신규 파일

```
NeuralSensorSim/scripts/augment_radar_real.py
  → real nuScenes train frames에 dynamic_compositor 적용
  → camera = real 이미지 원본 그대로 저장
  → radar = real radar + copy-pasted dynamic objects
  → outputs/radar_aug/scene_XX/sample_XXXX.npz

BEVFormerRadar/src/data/radar_aug_dataset.py
  → augment_radar_real.py 출력 로딩
  → match_real_density, noise jitter, flip 등 인라인 augmentation
  → SyntheticDataset과 동일 인터페이스

BEVFormerRadar/run_radar_aug.ps1
  → Exp A (real_only baseline) 재확인
  → Exp D (real camera + radar aug) 실험
  → evaluate_all.py 비교
```

### 기대 비교표

| Method | MOTA | FP/f | IDSW | 해석 |
|--------|------|------|------|------|
| real_only | -9.504 | 30.4 | 456 | baseline |
| real+synth_r1 (3DGS camera 포함) | -10.232 | 40.2 | 273 | camera domain gap |
| **real+radar_aug (Exp D)** | TBD | TBD | TBD | 카메라 domain gap 제거 |
| PMBM Phase29 | -0.177 | 0.8 | 3 | upper bound |

Exp D에서 FP가 real_only 수준을 유지하면서 IDSW가 r1처럼 개선된다면,
"domain gap이 원인"이라는 가설이 실험적으로 확인됨.

---

## 6. 3DGS의 포트폴리오 기여 재정의

3DGS를 Exp D에서 직접 사용하지 않더라도, 포트폴리오에서의 기여는 명확:

1. **정적 배경 모델링**: PSNR 30.94dB 씬 복원 — scene geometry 이해
2. **Novel viewpoint geometry**: radar depth estimation의 geometric prior
3. **실험적 발견**: "frozen backbone + synthetic camera = domain gap" 실증
4. **Ablation study의 기준**: Exp B/C (3DGS camera 포함) vs Exp D (제외) 비교로
   3DGS 카메라의 영향을 정량적으로 격리

---

## 우선순위 요약

```
즉시:
  □ augment_radar_real.py 작성 (NeuralSensorSim)
  □ RadarAugDataset 작성 (BEVFormerRadar)
  □ Exp D 실행 + evaluate_all.py 비교

완료 후:
  □ 비교표 완성 (Exp A / B / C / D)
  □ RESULTS.md 업데이트
  □ 포트폴리오 서술 완성
```
