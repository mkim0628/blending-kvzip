# ------------------------------------------------------------------------------
# Original Code developed by Jang-Hyun Kim
# GitHub Repository: https://github.com/snu-mllab/KVzip
# ------------------------------------------------------------------------------
import torch
import glob
from typing import List, Tuple, Union, Optional
from tqdm import tqdm
from transformers import DynamicCache, Gemma3ForCausalLM, Qwen3ForCausalLM

from attention.kvcache import RetainCache, EvictCache, RetainHybridCache
from utils.func import inplace_softmax
from model.load import load_model
from model.quant_model import OptimINT4KVCache, LlamaForCausalLMW8A8
from model.template import template


def chunk_fn(ctx_ids: torch.Tensor, chunk_size: int) -> List[torch.Tensor]:
    """ Chunk tokens
    """
    ctx_len = ctx_ids.shape[1]
    if ctx_len > chunk_size:
        chunk_num = (ctx_len - 1) // chunk_size + 1
        print(f"chunk inputs, size: {chunk_size} (num {chunk_num})")

        input_ids = []
        for i in range(chunk_num):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            a_ids = ctx_ids[:, start:end]
            if a_ids.shape[1] == 0:
                continue
            input_ids.append(a_ids)
    else:
        input_ids = [ctx_ids]

    return input_ids


def load_head_score(model_name, ctx_len):
    if model_name.startswith("Qwen2.5-7B"):
        model_name = "qwen2.5-7b"
    elif model_name.startswith("Qwen2.5-14B"):
        model_name = "qwen2.5-14b"
    elif model_name.startswith("Llama-3.1-8B"):
        model_name = "llama3.1-8b"

    attn_ = []
    paths = f"./utils/head_score/{model_name}-*.pt"
    for path in glob.glob(paths):
        attn = torch.load(path).squeeze().cuda()  # layer x head
        attn_.append(attn)
        print("Load head-score from", path)

    attn = torch.stack(attn_, dim=0).amax(0)
    score = attn.unsqueeze(-1).expand(-1, -1, ctx_len)  # layer x head x seq
    score = score.unsqueeze(1)
    return score


class ModelKVzip():

    def __init__(self, model_name: str, kv_type: str = "evict"):
        self.model, self.tokenizer = load_model(model_name)

        self.name = self.model.name
        self.dtype = self.model.dtype
        self.device = self.model.device
        self.config = self.model.config

        if isinstance(self.model, LlamaForCausalLMW8A8):
            self.kv_type = "int4static"
            print("[Note] Currently, only retain cache is available for QServe")
        elif isinstance(self.model, Gemma3ForCausalLM):
            self.kv_type = "hybrid_static"
            print("[Note] Currently, only retain cache is available for Gemma3")
        else:
            self.kv_type = kv_type
        print(f"KV type: {self.kv_type}")

        self.gen_kwargs = {
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1,
            "top_k": None,
            "max_new_tokens": 512,
        }
        if isinstance(self.model, Gemma3ForCausalLM):
            self.gen_kwargs["cache_implementation"] = None
            self.gen_kwargs["use_model_defaults"] = False
            self.gen_kwargs["eos_token_id"] = [1, 106]
        elif isinstance(self.model, Qwen3ForCausalLM):
            self.gen_kwargs["cache_implementation"] = None
            self.gen_kwargs["use_model_defaults"] = False
            self.gen_kwargs["eos_token_id"] = 151645

        self.set_chat_template()

    def encode(self, text: str) -> torch.Tensor:
        """ Encode text into tokens
        """
        return self.tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").cuda()

    def decode(self, input_ids: torch.Tensor) -> str:
        """ Decode tokens into text
        """
        if len(input_ids.shape) == 2:
            input_ids = input_ids[0]
        return self.tokenizer.decode(input_ids)

    def set_chat_template(self, task: str = "qa"):
        prefix, postfix = template(self.name, task)
        self.sys_prompt_ids, self.postfix_ids = self.encode(prefix), self.encode(postfix)

    def apply_template(self, query: str) -> torch.Tensor:
        query = f"\n\n{query.strip()}"
        query_ids = torch.cat([self.encode(query), self.postfix_ids], dim=1)
        return query_ids

    def __call__(
        self,
        input_ids: torch.Tensor,
        kv: Union[RetainCache, EvictCache],
        update_cache: bool = False,
        return_logits: bool = False,
        *args,
        **kwargs,
    ):
        """ Compute Transformer forward pass
            In default, we do not update the KV cache with the newly given inputs.
            Set update_cache = True to enable the update.
        """
        seen_token_prev = kv._seen_tokens

        if isinstance(kv, RetainHybridCache) and not update_cache:
            kv.backup_sliding_cache()

        if return_logits:
            outputs = self.model(input_ids, past_key_values=kv, *args, **kwargs)
        else:
            _ = self.model.model(input_ids, past_key_values=kv, *args, **kwargs)
            outputs = None

        if not update_cache:
            kv.slice(seen_token_prev)
        return outputs

    def _init_kv(self, kv=None, evict_range=(0, 0)):
        """ Initialize KV cache
        """

        if kv is None:
            if self.kv_type == "retain":
                kv = RetainCache(self.model, evict_range)
            elif self.kv_type == "evict":
                kv = EvictCache(self.model, evict_range)
            elif self.kv_type == "int4static":
                kv = OptimINT4KVCache(self.model.model, evict_range)
            elif self.kv_type == "hybrid_static":
                max_size = 190000
                kv = RetainHybridCache(self.model.model, evict_range, max_size)
            elif self.kv_type == "original":
                kv = DynamicCache()
                kv.pruned, kv.get_score = False, False
            else:
                raise NotImplementedError(f"type {self.kv_type} is not implemented")
        return kv

    @torch.inference_mode()
    def prefill(
        self,
        ctx_ids: Union[str, torch.Tensor],
        prefill_chunk_size: int = 16000,
        load_score=False,
        do_score=True,
    ) -> Union[RetainCache, EvictCache]:
        """ Chunked prefill KV cache
        """
        if type(ctx_ids) == str:
            ctx_ids = self.encode(ctx_ids)
        prefill_ids = torch.cat([self.sys_prompt_ids, ctx_ids], dim=1)
        evict_range = (self.sys_prompt_ids.shape[1], prefill_ids.shape[1])

        kv = self._init_kv(evict_range=evict_range)  # do not evict system prompt KV
        kv.ctx_ids = ctx_ids
        kv.prefill_ids = prefill_ids

        # prefill
        for input_ids in tqdm(chunk_fn(prefill_ids, prefill_chunk_size), desc="Prefill"):
            self.__call__(input_ids, kv, update_cache=True)

        if do_score:
            # KV importance scoring
            self.scoring(kv, ctx_ids, load_score=load_score)
        return kv

    def self_task(
        self,
        ctx_ids: torch.Tensor,
        chunk_size: int = 2000,
        prev_postfix_size=8,
    ) -> List[torch.Tensor]:
        """ Prepare chunked inputs for KV importance scoring with context reconstruction
            return: List[torch.Tensor]
        """
        chunked_inputs = chunk_fn(ctx_ids, chunk_size)

        input_ids = []
        for i, a_ids in enumerate(chunked_inputs):
            if i == 0:
                prompt = f"\n\nRepeat the previous context exactly."
                q_ids = self.encode(prompt)
            else:
                prompt = f"\n\nRepeat the part of the previous context exactly, starting with "
                q_ids = self.encode(prompt)
                postfix_prev = chunked_inputs[i - 1][:, -prev_postfix_size:]
                q_ids = torch.cat([q_ids, postfix_prev], dim=1)

            input_ids.append((a_ids, torch.cat([q_ids, self.postfix_ids, a_ids], dim=1)))

        return input_ids

    @torch.inference_mode()
    def scoring(
        self,
        kv: Union[RetainCache, EvictCache],
        ctx_ids: torch.Tensor,
        load_score=False,
    ):
        """ KV importance scoring (update kv.score)
        """
        if not load_score:
            kv.init_score()
            start_idx_tmp = kv.start_idx

            kv.end_idx = 0
            input_ids = self.self_task(ctx_ids)
            for i, (prefill_ids_p,
                    repeat_ids_p) in enumerate(tqdm(input_ids, desc=f"Importance scoring")):
                kv.end_idx = kv.start_idx + prefill_ids_p.shape[1]  # indices for a chunk
                self.__call__(repeat_ids_p, kv, update_cache=False)  # get score
                kv.start_idx = kv.end_idx

            kv.start_idx = start_idx_tmp
            assert kv.score[0].shape[-1] == kv.ctx_len
        else:
            kv.score = load_head_score(self.name, kv.ctx_len)

        kv.get_score = False

    @torch.inference_mode()
    def generate(
        self,
        query: Union[str, torch.Tensor],
        kv: Optional[Union[RetainCache, EvictCache]] = None,
        update_cache: bool = False,
    ) -> str:
        """ Obtain a model response to the query
            In default, we evict KV of query and generated answer after the generation by kv.slice (for multi-query evaluation).
            Set update_cache = True to enable multi-turn generation.
        """
        kv = self._init_kv(kv=kv)
        seen_token_prev = kv._seen_tokens

        if isinstance(kv, RetainHybridCache) and not update_cache:
            kv.backup_sliding_cache()

        input_ids = query
        if type(query) == str:
            input_ids = self.encode(query)
        if kv.prefill_ids is not None:
            # Huggingface Transformers model.generate requires full input tokens when using KV caches.
            # The inputs will be spliced to only contain new tokens as input[:, -kv.get_seq_length():].
            input_ids = torch.cat([kv.prefill_ids, input_ids], dim=1)

        output = self.model.generate(input_ids, past_key_values=kv, **self.gen_kwargs)
        a_ids = output[:, len(input_ids[0]):-1]  # parse response
        a = self.decode(a_ids)

        if not update_cache:
            kv.slice(seen_token_prev)
        else:
            kv.prefill_ids = torch.cat([input_ids, a_ids], dim=1)
        return a

    # ──────────────────────────────────────────────────────────
    # CacheBlend Integration (Compressed-State Blending)
    # ──────────────────────────────────────────────────────────

    @torch.inference_mode()
    def blend_generate(
        self,
        query: Union[str, torch.Tensor],
        chunk_kvs: list,
        recomp_ratio: float = 0.15,
        check_layers: list = None,
        update_cache: bool = False,
        position_offset: int = 0,
    ) -> str:
        """압축된 KV chunk를 blending하여 query에 응답.

        Compressed-state blending:
          1. RoPE 위치 보정 (position_offset > 0이면)
          2. Full forward로 fresh K/V 획득
          3. Check layer에서 head별로 flatten K와 fresh K 비교 (IW-HKVD)
          4. imp_indices를 flatten에 직접 overwrite
          5. 원본 EvictCache 객체로 generate → 압축 유지!

        Args:
            query: 사용자 쿼리 (str or tensor)
            chunk_kvs: ChunkStore.load_chunk()로 로드된 EvictCache 객체 리스트
            recomp_ratio: HKVD 재계산 비율
            check_layers: HKVD check layer (default: [1])
        """
        import time

        check_layers = check_layers or [1]
        device = self.device

        # 현재는 단일 chunk 지원 (multi-chunk는 추후 확장)
        kv = chunk_kvs[0]  # EvictCache 객체 그대로!
        n_layers = kv.n_layers
        n_heads_kv = kv.n_heads_kv
        is_pruned = getattr(kv, 'pruned', False)

        # ── 1. Full forward로 fresh K/V 획득 ──
        query_ids = query
        if type(query) == str:
            query_ids = self.encode(query)

        prefill_ids = kv.prefill_ids
        all_ids = torch.cat([prefill_ids, query_ids], dim=1)
        context_len = prefill_ids.shape[1]

        print(f"[Blend] Total: {all_ids.shape[1]} tokens "
              f"(context: {context_len}, query: {query_ids.shape[1]}, "
              f"position_offset: {position_offset})")

        # ── 1.5. RoPE 위치 보정 (position_offset > 0이면) ──
        if position_offset > 0 and is_pruned:
            from attention.blend import reapply_rope
            rotary_emb = self.model.model.rotary_emb
            t0 = time.perf_counter()

            for l in range(n_layers):
                cu_len_k = kv.info["cu_len_k"][l]
                valid_pad = torch.ones(1, n_heads_kv, kv.start_idx, dtype=torch.bool)
                full_valid = torch.cat([valid_pad, kv.valid[l].cpu()], dim=-1)

                for h in range(n_heads_kv):
                    kept_pos = full_valid[0, h].nonzero(as_tuple=True)[0]  # 원래 절대 위치
                    old_positions = kept_pos.to(device)
                    new_positions = (kept_pos + position_offset).to(device)

                    k_h = kv.key_cache[l][cu_len_k[h]:cu_len_k[h+1]]  # [len_k_h, D]
                    kv.key_cache[l][cu_len_k[h]:cu_len_k[h+1]] = \
                        reapply_rope(k_h, old_positions, new_positions, rotary_emb)

            t_rope = time.perf_counter() - t0
            print(f"[Blend] RoPE re-rotation: {t_rope*1000:.0f}ms (offset={position_offset})")

        # ── 1.6. Fresh forward ──
        fresh_cache = DynamicCache()
        t0 = time.perf_counter()
        self.model(all_ids, past_key_values=fresh_cache, use_cache=True)
        t_forward = time.perf_counter() - t0
        print(f"[Blend] Fresh forward: {t_forward*1000:.0f}ms")

        # ── 2. IW-HKVD ──
        is_pruned = getattr(kv, 'pruned', False)
        imp_per_head = {}

        if is_pruned:
            # Flatten 모드: head별로 flatten K와 fresh K 비교
            for cl in check_layers:
                info = kv.info
                cu_len_k = info["cu_len_k"][cl]
                valid_pad = torch.ones(1, n_heads_kv, kv.start_idx, dtype=torch.bool)
                full_valid = torch.cat([valid_pad, kv.valid[cl].cpu()], dim=-1)

                total_selected = 0
                for h in range(n_heads_kv):
                    kept_pos = full_valid[0, h].nonzero(as_tuple=True)[0].to(device)
                    k_old_h = kv.key_cache[cl][cu_len_k[h]:cu_len_k[h+1]]
                    k_new_h = fresh_cache.key_cache[cl][0, h, kept_pos, :]

                    diff_h = ((k_new_h - k_old_h) ** 2).sum(-1)

                    if kv.score is not None and cl < len(kv.score):
                        imp_score_h = kv.score[cl][0, h, :].to(device)
                        if imp_score_h.shape[0] < diff_h.shape[0]:
                            pad = torch.ones(diff_h.shape[0] - imp_score_h.shape[0], device=device)
                            imp_score_h = torch.cat([pad, imp_score_h])
                        elif imp_score_h.shape[0] > diff_h.shape[0]:
                            imp_score_h = imp_score_h[:diff_h.shape[0]]
                        diff_h = diff_h * imp_score_h

                    topk_h = max(int(len(diff_h) * recomp_ratio), 1)
                    imp_per_head[h] = torch.topk(diff_h, topk_h).indices
                    total_selected += len(imp_per_head[h])

                print(f"  [IW-HKVD] Layer {cl}: {total_selected} positions "
                      f"across {n_heads_kv} heads (r={recomp_ratio})")
        else:
            # Dense 모드: 무압축 KV, Phase 1 방식
            from attention.blend import iw_hkvd
            for cl in check_layers:
                k_old = kv.key_cache[cl]  # [1, H, T, D]
                k_new = fresh_cache.key_cache[cl]
                importance = kv.score[cl] if kv.score is not None and cl < len(kv.score) else None

                diff_k = ((k_new[:, :, :context_len].float() - k_old.float()) ** 2).sum(dim=[1, 3])
                if importance is not None:
                    imp_score = importance.float().mean(dim=1).to(device)
                    if imp_score.shape[-1] != diff_k.shape[-1]:
                        pad_len = diff_k.shape[-1] - imp_score.shape[-1]
                        if pad_len > 0:
                            imp_score = torch.cat([torch.ones(1, pad_len, device=device), imp_score], dim=-1)
                        else:
                            imp_score = imp_score[:, :diff_k.shape[-1]]
                    diff_k = diff_k * imp_score

                topk_num = max(int(context_len * recomp_ratio), 1)
                imp_indices = torch.topk(diff_k[0], topk_num).indices
                imp_indices, _ = torch.sort(imp_indices)

                # Dense overwrite
                kv.key_cache[cl][:, :, imp_indices] = k_new[:, :, imp_indices]
                kv.value_cache[cl][:, :, imp_indices] = k_new[:, :, imp_indices]

                print(f"  [IW-HKVD] Layer {cl}: {len(imp_indices)} tokens selected (dense, r={recomp_ratio})")

            # Non-check layers dense overwrite
            for l in range(n_layers):
                if l in check_layers:
                    continue
                kv.key_cache[l][:, :, imp_indices] = fresh_cache.key_cache[l][:, :, imp_indices]
                kv.value_cache[l][:, :, imp_indices] = fresh_cache.value_cache[l][:, :, imp_indices]

        # ── 3. Flatten overwrite (pruned only) ──
        if is_pruned:
            imp_abs_per_head = {}
            for cl in check_layers:
                valid_pad_cl = torch.ones(1, n_heads_kv, kv.start_idx, dtype=torch.bool)
                full_valid_cl = torch.cat([valid_pad_cl, kv.valid[cl].cpu()], dim=-1)
                for h in range(n_heads_kv):
                    kept_pos_cl = full_valid_cl[0, h].nonzero(as_tuple=True)[0]
                    imp_abs_per_head[h] = set(kept_pos_cl[imp_per_head[h].cpu()].tolist())

            t0 = time.perf_counter()
            for l in range(n_layers):
                cu_len_k = kv.info["cu_len_k"][l]
                valid_pad = torch.ones(1, n_heads_kv, kv.start_idx, dtype=torch.bool)
                full_valid = torch.cat([valid_pad, kv.valid[l].cpu()], dim=-1)

                for h in range(n_heads_kv):
                    if h not in imp_abs_per_head:
                        continue
                    kept_pos = full_valid[0, h].nonzero(as_tuple=True)[0].to(device)
                    abs_targets = imp_abs_per_head[h]

                    local_imp = [i for i, p in enumerate(kept_pos.tolist()) if p in abs_targets]
                    if not local_imp:
                        continue

                    local_imp_t = torch.tensor(local_imp, device=device, dtype=torch.long)
                    abs_positions = kept_pos[local_imp_t]

                    kv.key_cache[l][cu_len_k[h] + local_imp_t] = \
                        fresh_cache.key_cache[l][0, h, abs_positions, :]
                    kv.value_cache[l][cu_len_k[h] + local_imp_t] = \
                        fresh_cache.value_cache[l][0, h, abs_positions, :]

            t_blend = time.perf_counter() - t0
            print(f"[Blend] Overwrite: {t_blend*1000:.0f}ms")
        else:
            print(f"[Blend] Dense overwrite done (in step 2)")

        # ── 4. Generate (원본 EvictCache 객체 그대로 사용!) ──
        t0 = time.perf_counter()
        output = self.generate(query_ids, kv=kv, update_cache=False)
        t_gen = time.perf_counter() - t0
        print(f"[Blend] Generate: {t_gen*1000:.0f}ms, output: {output[:100]}...")

        return output

    @torch.inference_mode()
    def blend_generate_multi(
        self,
        query: Union[str, torch.Tensor],
        chunk_kvs: list,
        recomp_ratio: float = 0.15,
        check_layers: list = None,
        method: str = "iw_hkvd",
    ) -> str:
        """여러 압축 chunk를 순서대로 합쳐서 blending + generate.

        Multi-chunk 시나리오:
          오프라인: doc_A, doc_B 각각 따로 저장
          온라인:   [doc_A + doc_B + query] 또는 [doc_B + doc_A + query]

        각 chunk의 RoPE를 새 위치에 맞게 보정하고, 하나의 EvictCache로 합친 뒤
        IW-HKVD blend + generate.

        Args:
            query: 사용자 쿼리
            chunk_kvs: EvictCache 객체 리스트 (순서대로 배치)
            recomp_ratio: 재계산 비율
            check_layers: check layer
            method: "iw_hkvd" | "diff_only" | "random"
        """
        import time
        from attention.blend import reapply_rope
        from transformers import DynamicCache

        check_layers = check_layers or [1]
        device = self.device
        n_chunks = len(chunk_kvs)

        kv_first = chunk_kvs[0]
        n_layers = kv_first.n_layers
        n_heads_kv = kv_first.n_heads_kv

        # ── 1. 전체 prompt 구성 + 각 chunk의 position offset 계산 ──
        query_ids = query
        if type(query) == str:
            query_ids = self.encode(query)

        all_prefill = []
        chunk_offsets = []  # 각 chunk가 전체 prompt에서 시작하는 position
        pos = 0
        for ci, kv_c in enumerate(chunk_kvs):
            chunk_offsets.append(pos)
            all_prefill.append(kv_c.prefill_ids)
            pos += kv_c.prefill_ids.shape[1]

        prefill_ids = torch.cat(all_prefill, dim=1)
        all_ids = torch.cat([prefill_ids, query_ids], dim=1)
        context_len = prefill_ids.shape[1]

        print(f"[MultiBlend] {n_chunks} chunks, total: {all_ids.shape[1]} tokens "
              f"(context: {context_len}, query: {query_ids.shape[1]})")
        for ci, kv_c in enumerate(chunk_kvs):
            print(f"  chunk {ci}: {kv_c._seen_tokens} tokens, offset={chunk_offsets[ci]}")
            # info의 tensor들을 GPU로 이동
            for l in range(n_layers):
                if isinstance(kv_c.info["cu_len_k"][l], torch.Tensor):
                    kv_c.info["cu_len_k"][l] = kv_c.info["cu_len_k"][l].to(device)
                if isinstance(kv_c.info["len_k"][l], torch.Tensor):
                    kv_c.info["len_k"][l] = kv_c.info["len_k"][l].to(device)
            if isinstance(kv_c.info.get("cu_head"), torch.Tensor):
                kv_c.info["cu_head"] = kv_c.info["cu_head"].to(device)
            # valid도 GPU로
            if hasattr(kv_c, 'valid') and kv_c.valid is not None:
                kv_c.valid = kv_c.valid.to(device)
            if hasattr(kv_c, 'valid_pad') and kv_c.valid_pad is not None:
                kv_c.valid_pad = kv_c.valid_pad.to(device)

        # ── 2. 각 chunk의 RoPE 보정 ──
        rotary_emb = self.model.model.rotary_emb
        t0 = time.perf_counter()

        for ci, kv_c in enumerate(chunk_kvs):
            offset = chunk_offsets[ci]
            if offset == 0 and ci == 0:
                continue  # 첫 chunk는 position 0이라 보정 불필요 (원래 위치 그대로)

            for l in range(n_layers):
                cu_len_k = kv_c.info["cu_len_k"][l]
                valid_pad = torch.ones(1, n_heads_kv, kv_c.start_idx, dtype=torch.bool, device=device)
                full_valid = torch.cat([valid_pad, kv_c.valid[l]], dim=-1)

                for h in range(n_heads_kv):
                    kept_pos = full_valid[0, h].nonzero(as_tuple=True)[0]
                    old_positions = kept_pos.to(device)
                    new_positions = (kept_pos + offset).to(device)
                    k_h = kv_c.key_cache[l][cu_len_k[h]:cu_len_k[h+1]]
                    kv_c.key_cache[l][cu_len_k[h]:cu_len_k[h+1]] = \
                        reapply_rope(k_h, old_positions, new_positions, rotary_emb)

        t_rope = time.perf_counter() - t0
        print(f"[MultiBlend] RoPE re-rotation: {t_rope*1000:.0f}ms")

        # ── 3. Fresh forward ──
        fresh_cache = DynamicCache()
        t0 = time.perf_counter()
        self.model(all_ids, past_key_values=fresh_cache, use_cache=True)
        t_forward = time.perf_counter() - t0
        print(f"[MultiBlend] Fresh forward: {t_forward*1000:.0f}ms")

        # ── 4. 각 chunk에서 IW-HKVD + overwrite ──
        t0 = time.perf_counter()
        total_selected = 0

        for ci, kv_c in enumerate(chunk_kvs):
            offset = chunk_offsets[ci]

            for cl in check_layers:
                cu_len_k = kv_c.info["cu_len_k"][cl]
                valid_pad = torch.ones(1, n_heads_kv, kv_c.start_idx, dtype=torch.bool, device=device)
                full_valid = torch.cat([valid_pad, kv_c.valid[cl]], dim=-1)

                imp_per_head = {}
                for h in range(n_heads_kv):
                    kept_pos = full_valid[0, h].nonzero(as_tuple=True)[0]
                    abs_kept = (kept_pos + offset).to(device)

                    k_old_h = kv_c.key_cache[cl][cu_len_k[h]:cu_len_k[h+1]]
                    k_new_h = fresh_cache.key_cache[cl][0, h, abs_kept, :]

                    diff_h = ((k_new_h - k_old_h) ** 2).sum(-1)
                    len_k_h = k_old_h.shape[0]
                    topk_h = max(int(len_k_h * recomp_ratio), 1)

                    if method == "iw_hkvd" and kv_c.score is not None and cl < len(kv_c.score):
                        imp_score_h = kv_c.score[cl][0, h, :].to(device)
                        if imp_score_h.shape[0] < len_k_h:
                            pad = torch.ones(len_k_h - imp_score_h.shape[0], device=device)
                            imp_score_h = torch.cat([pad, imp_score_h])
                        elif imp_score_h.shape[0] > len_k_h:
                            imp_score_h = imp_score_h[:len_k_h]
                        metric = diff_h * imp_score_h
                    elif method == "random":
                        metric = torch.rand(len_k_h, device=device)
                    else:
                        metric = diff_h

                    imp_per_head[h] = torch.topk(metric, topk_h).indices
                    total_selected += topk_h

                # Overwrite all layers for this chunk
                imp_abs_per_head = {}
                valid_pad_cl = torch.ones(1, n_heads_kv, kv_c.start_idx, dtype=torch.bool, device=device)
                full_valid_cl = torch.cat([valid_pad_cl, kv_c.valid[cl]], dim=-1)
                for h in range(n_heads_kv):
                    kept_pos_cl = full_valid_cl[0, h].nonzero(as_tuple=True)[0]
                    imp_abs_per_head[h] = set(kept_pos_cl[imp_per_head[h].cpu()].tolist())

                for l in range(n_layers):
                    cu_len_k_l = kv_c.info["cu_len_k"][l]
                    valid_pad_l = torch.ones(1, n_heads_kv, kv_c.start_idx, dtype=torch.bool, device=device)
                    full_valid_l = torch.cat([valid_pad_l, kv_c.valid[l]], dim=-1)

                    for h in range(n_heads_kv):
                        if h not in imp_abs_per_head:
                            continue
                        kept_pos_l = full_valid_l[0, h].nonzero(as_tuple=True)[0]
                        abs_targets = imp_abs_per_head[h]
                        local_imp = [i for i, p in enumerate(kept_pos_l.tolist()) if p in abs_targets]
                        if not local_imp:
                            continue
                        lt = torch.tensor(local_imp, dtype=torch.long).to(device)
                        abs_pos = (kept_pos_l[lt] + offset).to(device)
                        kv_c.key_cache[l][cu_len_k_l[h] + lt] = \
                            fresh_cache.key_cache[l][0, h, abs_pos, :]
                        kv_c.value_cache[l][cu_len_k_l[h] + lt] = \
                            fresh_cache.value_cache[l][0, h, abs_pos, :]

        t_blend = time.perf_counter() - t0
        print(f"[MultiBlend] HKVD+Overwrite: {t_blend*1000:.0f}ms ({total_selected} positions, method={method})")

        # ── 5. Merge chunks into one EvictCache for decode ──
        # head별로 두 chunk의 토큰을 합침
        # 올바른 순서: [c1_h0 + c2_h0, c1_h1 + c2_h1, ...] (head별 interleave)
        kv_merged = chunk_kvs[0]

        for ci in range(1, n_chunks):
            kv_next = chunk_kvs[ci]
            for l in range(n_layers):
                # head별로 interleave merge
                merged_k_parts = []
                merged_v_parts = []
                new_cu = [torch.tensor([0], dtype=torch.int32, device=device)]
                new_lens = []
                running_offset = 0

                cu_m = kv_merged.info["cu_len_k"][l]
                cu_n = kv_next.info["cu_len_k"][l]

                for h in range(n_heads_kv):
                    # 기존 merged의 head h
                    k_m_h = kv_merged.key_cache[l][cu_m[h]:cu_m[h+1]]
                    v_m_h = kv_merged.value_cache[l][cu_m[h]:cu_m[h+1]]
                    # 새 chunk의 head h
                    k_n_h = kv_next.key_cache[l][cu_n[h]:cu_n[h+1]]
                    v_n_h = kv_next.value_cache[l][cu_n[h]:cu_n[h+1]]
                    # 합치기
                    merged_k_parts.append(torch.cat([k_m_h, k_n_h], dim=0))
                    merged_v_parts.append(torch.cat([v_m_h, v_n_h], dim=0))

                    head_len = k_m_h.shape[0] + k_n_h.shape[0]
                    running_offset += head_len
                    new_cu.append(torch.tensor([running_offset], dtype=torch.int32, device=device))
                    new_lens.append(head_len)

                kv_merged.key_cache[l] = torch.cat(merged_k_parts, dim=0)
                kv_merged.value_cache[l] = torch.cat(merged_v_parts, dim=0)
                kv_merged.info["cu_len_k"][l] = torch.cat(new_cu)
                kv_merged.info["len_k"][l] = torch.tensor(new_lens, dtype=torch.int32, device=device)
                kv_merged.info["max_len_k"][l] = max(new_lens)

        kv_merged.prefill_ids = prefill_ids
        kv_merged._seen_tokens = context_len
        kv_merged.info["cu_head"] = torch.arange(n_heads_kv + 1, dtype=torch.int32, device=device)

        print(f"[MultiBlend] Merged: {kv_merged.key_cache[0].shape[0]} flatten tokens, "
              f"cu_head={kv_merged.info['cu_head'].tolist()}")

        # ── 6. Generate ──
        t0 = time.perf_counter()
        output = self.generate(query_ids, kv=kv_merged, update_cache=False)
        t_gen = time.perf_counter() - t0
        print(f"[MultiBlend] Generate: {t_gen*1000:.0f}ms, output: {output[:100]}...")

        return output

    @torch.inference_mode()
    def _prob(self, input_ids, kv=None, device="cuda") -> torch.Tensor:
        """ Obtain next token prediction probabilities
        """
        kv = self._init_kv(kv=kv)

        if isinstance(self.model, LlamaForCausalLMW8A8):
            output = self.__call__(input_ids,
                                   kv,
                                   update_cache=False,
                                   return_logits=True,
                                   is_prompt=False)
            output = output[0]
        else:
            output = self.__call__(input_ids, kv, update_cache=False, return_logits=True)
            output = output.logits[0]
        output = inplace_softmax(output).squeeze()

        if device == "cpu":
            return output.cpu()
        return output
