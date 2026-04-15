"""IW-HKVD: Importance-Weighted High KV Deviation blending.

CacheBlend의 "변화도" × KVzip의 "중요도"를 결합하여
"변했고 중요한" 토큰만 재계산한다.

KVzip의 기존 코드와 함께 사용:
  1. prefill + scoring + prune → EvictCache에 압축 KV + importance score 저장
  2. 새 query 시 → blend()로 HKVD 수행 → generate
"""

import os
import hashlib
import torch
from typing import Dict, List, Optional, Tuple, Union
from transformers import DynamicCache
from attention.kvcache import EvictCache


# ──────────────────────────────────────────────────────────────
# 1. ChunkStore: 압축 KV를 디스크에 저장/로드
# ──────────────────────────────────────────────────────────────

class ChunkStore:
    """KVzip으로 압축된 KV chunk를 디스크에 저장/로드."""

    def __init__(self, store_dir: str = "./chunk_store"):
        self.store_dir = store_dir
        os.makedirs(store_dir, exist_ok=True)

    def save_chunk(self, chunk_id: str, kv: EvictCache):
        """EvictCache 객체를 통째로 저장.

        Args:
            chunk_id: 청크 식별자 (e.g., "doc_1")
            kv: KVzip prefill + scoring + prune이 완료된 EvictCache
        """
        path = os.path.join(self.store_dir, f"{chunk_id}.pt")
        torch.save(kv, path)
        print(f"[ChunkStore] Saved '{chunk_id}' ({kv._seen_tokens} tokens, pruned={kv.pruned})")

    def load_chunk(self, chunk_id: str, device: str = "cuda") -> EvictCache:
        """저장된 EvictCache 객체를 로드."""
        path = os.path.join(self.store_dir, f"{chunk_id}.pt")
        kv = torch.load(path, map_location=device, weights_only=False)
        print(f"[ChunkStore] Loaded '{chunk_id}' ({kv._seen_tokens} tokens, pruned={kv.pruned})")
        return kv

    def list_chunks(self) -> List[str]:
        return sorted([f[:-3] for f in os.listdir(self.store_dir) if f.endswith(".pt")])


# ──────────────────────────────────────────────────────────────
# 2. IW-HKVD: 핵심 알고리즘
# ──────────────────────────────────────────────────────────────

def iw_hkvd(
    k_new: torch.Tensor,
    k_old: torch.Tensor,
    recomp_ratio: float,
    importance: Optional[torch.Tensor] = None,
    mandatory: Optional[List[int]] = None,
) -> torch.Tensor:
    """Importance-Weighted HKVD — 재계산할 토큰 선택.

    Args:
        k_new: [1, H_kv, T_total, D] — fresh K (post-RoPE)
        k_old: [1, H_kv, T_cached, D] — cached K (post-RoPE)
        recomp_ratio: 재계산 비율
        importance: [1, H_kv, T_cached] — KVzip score (optional)
        mandatory: sink + boundary 인덱스 (optional)

    Returns:
        imp_indices: [N] — 재계산할 토큰 인덱스 (sorted, unique)
    """
    T_cached = k_old.shape[2]
    device = k_old.device

    # 변화도
    diff_k = ((k_new[:, :, :T_cached].float() - k_old.float()) ** 2).sum(dim=[1, 3])  # [1, T]

    # 중요도 가중
    if importance is not None:
        imp_score = importance.float().mean(dim=1)  # [1, T_imp] — head 평균
        # importance 크기 맞추기 (scoring은 context만, KV는 sys+context 포함 가능)
        if imp_score.shape[-1] != T_cached:
            pad_len = T_cached - imp_score.shape[-1]
            if pad_len > 0:
                pad = torch.ones(1, pad_len, device=device, dtype=imp_score.dtype)
                imp_score = torch.cat([pad, imp_score], dim=-1)
            else:
                imp_score = imp_score[:, :T_cached]
        weighted_diff = diff_k * imp_score
    else:
        weighted_diff = diff_k

    # mandatory 처리
    mandatory = mandatory or []
    total_budget = max(int(T_cached * recomp_ratio), 1)
    elective_budget = max(total_budget - len(mandatory), 0)

    if mandatory:
        mandatory_t = torch.tensor(mandatory, device=device, dtype=torch.long)
        weighted_diff[0, mandatory_t] = -1.0

    # top-k 선택
    if elective_budget > 0:
        elective = torch.topk(weighted_diff[0], min(elective_budget, T_cached)).indices
    else:
        elective = torch.tensor([], device=device, dtype=torch.long)

    # 합치기
    if mandatory:
        all_idx = torch.cat([mandatory_t, elective])
    else:
        all_idx = elective

    return torch.unique(torch.sort(all_idx)[0])


# ──────────────────────────────────────────────────────────────
# 3. Blend attention hook (_blend 함수)
# ──────────────────────────────────────────────────────────────

def blend_hook(
    past_key_value,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_idx: int,
):
    """attn.py에서 호출되는 blend hook.

    past_key_value가 blending=True이면 IW-HKVD를 수행한다.

    Args:
        past_key_value: BlendEvictCache (blending + pruned 상태)
        query_states: [1, H, T, D]
        key_states:   [1, H_kv, T, D] — fresh K (post-RoPE)
        value_states: [1, H_kv, T, D] — fresh V
        layer_idx: int

    Returns:
        query_states, key_states, value_states (blended)
    """
    kv = past_key_value
    cached_len = kv.blend_cached_len

    k_old = kv.key_cache[layer_idx]   # [1, H_kv, T_cached, D]
    v_old = kv.value_cache[layer_idx]

    # Check layer: IW-HKVD
    if layer_idx in kv.blend_check_layers:
        importance = kv.blend_importance.get(layer_idx, None)

        kv.blend_imp_indices = iw_hkvd(
            k_new=key_states,
            k_old=k_old,
            recomp_ratio=kv.blend_recomp_ratio,
            importance=importance,
            mandatory=kv.blend_mandatory,
        )

        n_selected = len(kv.blend_imp_indices)
        print(f"  [IW-HKVD] Layer {layer_idx}: {n_selected}/{cached_len} tokens selected "
              f"(r={kv.blend_recomp_ratio})")

    # 선택된 위치를 fresh K/V로 덮어쓰기
    if kv.blend_imp_indices is not None and len(kv.blend_imp_indices) > 0:
        imp = kv.blend_imp_indices
        k_old[:, :, imp] = key_states[:, :, imp]
        v_old[:, :, imp] = value_states[:, :, imp]

    # query 토큰(캐시에 없는 부분) 추가
    if key_states.shape[2] > cached_len:
        k_full = torch.cat([k_old, key_states[:, :, cached_len:]], dim=2)
        v_full = torch.cat([v_old, value_states[:, :, cached_len:]], dim=2)
    else:
        k_full = k_old
        v_full = v_old

    # cache 갱신
    kv.key_cache[layer_idx] = k_full
    kv.value_cache[layer_idx] = v_full

    return query_states, k_full, v_full
