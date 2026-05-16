"""
MixedDataset — Real(nuScenes) + Synthetic(.npz) 데이터셋 혼합.

ratio: synthetic / real 비율
  1.0 → 1:1  (real 242 + synth 242)
  3.0 → 1:3  (real 242 + synth 726)
  0.0 → real only (baseline)

샘플 순서: real 시퀀스를 유지한 채 synthetic을 중간에 삽입.
(temporal state는 prev_token="" 기준으로 자동 초기화)
"""

import numpy as np
from torch.utils.data import Dataset, ConcatDataset


class MixedDataset(Dataset):
    """
    Args:
        real_ds:   BEVFormerDataset (shuffle=False 권장 — temporal 순서 유지)
        synth_ds:  SyntheticDataset
        ratio:     synthetic 비율 (synth 샘플 수 = int(len(real_ds) * ratio))
        seed:      샘플링 재현성용 시드
    """

    def __init__(self, real_ds: Dataset, synth_ds: Dataset,
                 ratio: float = 1.0, seed: int = 42):
        rng = np.random.default_rng(seed)

        n_synth = min(int(len(real_ds) * ratio), len(synth_ds))

        # synthetic 인덱스 샘플링
        synth_indices = rng.choice(len(synth_ds), size=n_synth, replace=False)

        # 인덱스 목록: (dataset, local_idx)
        self._items = (
            [("real",  i) for i in range(len(real_ds))] +
            [("synth", int(i)) for i in synth_indices]
        )

        # 순서: real 먼저, synthetic 뒤에 (또는 섞기)
        # temporal 연속성 때문에 real 블록을 먼저 유지
        self._real_ds  = real_ds
        self._synth_ds = synth_ds

        print(f"[MixedDataset] real={len(real_ds)}  synthetic={n_synth}  "
              f"total={len(self._items)}  ratio=1:{ratio:.1f}")

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        source, local_idx = self._items[idx]
        if source == "real":
            return self._real_ds[local_idx]
        else:
            return self._synth_ds[local_idx]

    @property
    def n_real(self) -> int:
        return sum(1 for s, _ in self._items if s == "real")

    @property
    def n_synth(self) -> int:
        return sum(1 for s, _ in self._items if s == "synth")
