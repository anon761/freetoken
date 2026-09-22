"""MTP speculative-decode rounds (greedy + sampled; overlap integrated).

One round per scheduler iteration. The round is split so ``overlap_loop`` can make the
verify the overlapped batch (Phase 4):

* ``begin_round`` -- grow the ids / allocate the verify window, run the draft chain,
  snapshot the GDN state and LAUNCH the verify asynchronously;
* drain the previous batch on the host (this is the overlap window);
* ``finish_round`` -- accept, publish, roll back rejected pages and commit the accepted
  prefix into the live conv + SSM slots.

``normal_loop`` calls ``_round`` (begin+finish back-to-back, the pre-Phase-4 behavior).

Round shape:

1. Entry state per request (the invariant every engine forward leaves behind):
   ``cached_len == device_len - 1`` with the freshly sampled token pending at
   position ``C = cached_len`` (staged in ``token_pool`` and appended on host).
2. Draft chain: the MTP block runs k sequential single-token forwards, each
   consuming ``(R_last, token)`` and predicting the next draft id. Draft KV lands
   in the MTP block's own QSA slab (``layer_id = num_layers``) at the REAL main
   positions -- no conflict with the main layers, which only the verify forward
   writes.
3. Verify: ONE prefill-phase extend over ``[C, C+k+1)`` (pending token + k drafts)
   through the full model. Greedy acceptance: the argmax at position ``C+j`` must
   equal draft ``d_j``. Sampled acceptance: vLLM-style rejection sampling against the
   target's truncated distribution (the greedy draft is a point-mass proposal). The
   bonus token is the argmax at the last accepted position (greedy) or a sample from
   the target / residual distribution (sampled).
4. Publish accepted drafts + bonus (EOS / stop-string / length checks per token),
   roll back the rejected pages (``CacheManager.rollback_last``), restore the GDN
   state to the pre-verify boundary and commit the accepted prefix into the live
   conv + SSM slots (``commit_mtp_verify``, Phase 2 -- no full-model re-extend), so
   the next round starts from a clean, uniform entry state.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, NamedTuple, Optional, Sequence, Set, Tuple

import torch
from freetoken.core import Batch, Req
from freetoken.engine.sample import probs_from_logits, sample_residual
from freetoken.env import ENV
from freetoken.message import DetokenizeMsg
from freetoken.utils import init_logger

from .ngram_draft import ngram_draft

if TYPE_CHECKING:
    from .scheduler import Scheduler

logger = init_logger(__name__)

# The GDN chunk kernel only materializes state at x64 boundaries; a verify extend
# must stay inside ONE chunk so the in-place state advance is confined to a single
# snapshot/restore window (see _GDN_CHUNK in attention/linear.py).
_GDN_CHUNK = 64


class _DraftReq:
    """Minimal request view for synthetic draft batches. The QSA backend's metadata
    reads only ``table_idx`` / ``cached_len`` / ``device_len`` / ``extend_len`` off
    ``padded_reqs``; everything else (positions, out_loc) rides on the batch."""

    __slots__ = ("table_idx", "cached_len", "device_len")

    def __init__(self, table_idx: int, cached_len: int, device_len: int) -> None:
        self.table_idx = table_idx
        self.cached_len = cached_len
        self.device_len = device_len

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len


class _SynthBatch:
    """Synthetic prefill-phase batch for MTP draft forwards: the attribute surface
    the QSA backend and lm_head read off a real Batch. Ragged metadata (phase
    "prefill") so any token count per request is addressed correctly; never touches
    the shared decode-graph staging buffers."""

    def __init__(
        self,
        reqs: Sequence[_DraftReq],
        positions: torch.Tensor,
        out_loc: torch.Tensor,
    ) -> None:
        self.reqs: List[_DraftReq] = list(reqs)
        self.positions = positions
        self.out_loc = out_loc
        self.phase = "prefill"
        self.attn_metadata = None

    @property
    def padded_reqs(self) -> List[_DraftReq]:
        return self.reqs

    @property
    def is_prefill(self) -> bool:
        return True

    @property
    def is_decode(self) -> bool:
        return False

    @property
    def size(self) -> int:
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        return len(self.reqs)


class _RoundState(NamedTuple):
    """Handoff from ``begin_round`` (draft + verify, verify launched async) to
    ``finish_round`` (accept + publish + rollback + GDN commit). Splitting the round lets
    ``overlap_loop`` drain the previous batch on the host while the verify runs; the two
    halves are otherwise one synchronous round (``run_round``)."""

    reqs: List[Req]
    C: List[int]
    k: int
    n: int
    dev: torch.device
    verify_batch: Batch
    logits: torch.Tensor
    verify_streams: Optional[torch.Tensor]
    drafts_gpu: List[torch.Tensor]
    anchor_toks: torch.Tensor
    hybrid: bool
    scratch: List[int]
    pool: object
    moe: object
    timing: bool


class MTPManager:
    # qwen4-exp MTP carries per-request residual streams (`req.mtp_streams_row`) that the
    # scheduler must refresh every decode step; DSpark does not (uses the target aux buffer).
    uses_streams = True

    def __init__(self, sched: "Scheduler") -> None:
        self.sched = sched
        self.engine = sched.engine
        model = sched.engine.model
        self.mtp = getattr(model, "mtp", None)
        self.enabled = self.mtp is not None
        self._deferred: set[Req] = set()
        if not self.enabled:
            self.k = 0
            logger.info_rank0(
                "MTP speculative decoding disabled (no draft head built; --mtp off/auto)"
            )
            return
        self.k = max(1, ENV.MTP_DRAFT_TOKENS.value)
        assert self.k + 1 < _GDN_CHUNK, (
            f"MTP verify extends must stay inside one GDN chunk ({_GDN_CHUNK})"
        )
        self.embed = model.model.embed_tokens.forward
        self.lm_head = model.lm_head
        # Stream width of the draft input: qwen4-exp carries hyper-connection streams
        # (ple_state_width); the dense qwen3_5 head consumes the plain hidden.
        qwen4 = getattr(sched.config.model_config, "qwen4_args", None)
        self.width = getattr(qwen4, "ple_state_width", None) or sched.config.model_config.hidden_size
        # Cumulative acceptance counters for /v1/stats (last-known-wins via the reply
        # stamps): drafted = k per round per request, accepted = verified draft tokens.
        self.stats_drafted = 0
        self.stats_accepted = 0
        self.stats_rounds = 0
        self.stats_rounds_logged = 0
        # n-gram combiner (self-history draft): override a request's MTP draft chain when
        # the continuation after its final NGRAM_SIZE tokens repeats earlier in its history.
        self.ngram_enabled = ENV.MTP_NGRAM
        self.ngram_size = max(1, ENV.MTP_NGRAM_SIZE.value)
        self.stats_ngram_rounds = 0
        # Per-draft-index match counts (device-resident; see _round). Read out in the log.
        self.stats_match = torch.zeros(self.k, dtype=torch.int64, device=sched.engine.device)
        # Histogram of the accepted-draft count a (host): P(a>=j) is the conditional
        # acceptance curve -- the signal for k tuning vs. a speculative token tree.
        self.a_hist = [0] * (self.k + 1)
        # Per-phase round timing (FREETOKEN_MTP_TIMING=1), accumulated then averaged.
        self._timing_acc: dict[str, float] = {}
        self._timing_n = 0
        self._timing_on = os.environ.get("FREETOKEN_MTP_TIMING") == "1"
        moe = getattr(sched.engine, "moe_offload_cache", None)
        logger.info_rank0(
            f"MTP speculative decoding enabled: k={self.k}, eager verify, "
            f"greedy{' + sampled' if ENV.MTP_SAMPLED else ''} | MoE "
            f"strategy={getattr(moe, 'decode_target', '?')} "
            f"verify_cpu={getattr(moe, 'verify_cpu', False)}"
        )

    # ------------------------------------------------------------------ helpers

    def eligible(self, req: Req) -> bool:
        return (
            self.enabled
            and req.table_idx != -1
            and not req.aborted
            and req.mm_embeds is None
            and req.mtp_streams_row is not None
            # Greedy always; sampled only with FREETOKEN_MTP_SAMPLED=1 (Phase 1 is
            # correct but currently a net slowdown, so it is opt-in).
            and (req.sampling_params.is_greedy or ENV.MTP_SAMPLED)
            # room for k drafts + the bonus token within max_device_len
            and req.remain_len >= self.k + 1
        )

    def _synth_batch(
        self, reqs: Sequence[_DraftReq], positions: torch.Tensor, out_loc: torch.Tensor
    ) -> _SynthBatch:
        sb = _SynthBatch(reqs, positions, out_loc)
        self.engine.attn_backend.prepare_metadata(sb)
        return sb

    # ------------------------------------------------------- prefill-side hook

    def draft_prefill(self, batch: Batch, pre_lens: List[Tuple[int, int]]) -> None:
        """Build the MTP block's own KV over a (chunked) prefill extend, right after
        the main model's forward. One ragged draft forward over all of the batch's
        extends: the draft row for position p fuses the MAIN streams of position p-1
        (the saved row across a chunk boundary) with token p's embedding.

        ``pre_lens`` is the per-req ``(cached_len, device_len)`` snapshot taken BEFORE
        the forward — complete_one has already advanced both by the time this runs, so
        the extend range is not derivable from the req anymore."""
        if not self.enabled:
            return
        streams = batch.mtp_streams
        if streams is None:  # CUDA-graph replay or non-stashing model
            return
        eng = self.engine
        rows: List[torch.Tensor] = []
        toks: List[int] = []
        seg_reqs: List[_DraftReq] = []
        offsets = torch.zeros(len(batch.reqs) + 1, dtype=torch.int64)
        zero_row = torch.zeros(self.width, dtype=streams.dtype, device=eng.device)
        for i, (r, (s, e)) in enumerate(zip(batch.reqs, pre_lens)):
            if r.mm_embeds is not None or r.table_idx == -1:
                offsets[i + 1] = offsets[i]
                continue
            if e <= s:
                offsets[i + 1] = offsets[i]
                continue
            seg = streams[offsets[i] : offsets[i] + (e - s)]
            for j, p in enumerate(range(s, e)):
                if p == 0:
                    rows.append(zero_row)  # fresh sequence: no predecessor streams
                elif j == 0:
                    # chunk boundary: the predecessor position lives in the previous
                    # chunk's forward -- its streams were saved on the request
                    rows.append(r.mtp_streams_row if r.mtp_streams_row is not None else zero_row)
                else:
                    rows.append(seg[j - 1])
                toks.append(int(r.input_ids[p]))
            seg_reqs.append(_DraftReq(r.table_idx, s, e))
            r.mtp_streams_row = seg[-1].clone()
            offsets[i + 1] = offsets[i] + (e - s)
        if not seg_reqs:
            return
        tables = torch.tensor([q.table_idx for q in seg_reqs], dtype=torch.int64, device=eng.device)
        seg_lens = torch.tensor([q.extend_len for q in seg_reqs], device=eng.device)
        pos = torch.tensor(
            [q.cached_len + j for q in seg_reqs for j in range(q.extend_len)],
            dtype=torch.int32,
            device=eng.device,
        )
        out_loc = eng.page_table[
            torch.repeat_interleave(tables, seg_lens), pos.to(torch.int64)
        ]
        R_last = torch.stack(rows, dim=0)
        tokens = torch.tensor(toks, dtype=torch.int32, device=eng.device)
        sb = self._synth_batch(seg_reqs, pos, out_loc)
        with eng.ctx.forward_batch(sb):
            self.mtp.draft_step(self.embed, R_last, tokens, sb)

    # ----------------------------------------------------------- the round

    def run_round(self) -> None:
        """At most one speculative round per scheduler iteration, before decode
        scheduling. Prefill priority: while a prefill is pending, this iteration's
        eligible requests are DEFERRED (their pending token waits; the decode batch
        excludes them) so a graph-replayed plain decode step cannot knock them out
        of MTP by clearing their streams."""
        if not self.enabled:
            return
        if self.sched.prefill_manager.runnable:
            self._deferred = {
                r for r in self.sched.decode_manager.running_reqs if self.eligible(r)
            }
            return
        self._deferred = set()
        reqs = [r for r in self.sched.decode_manager.running_reqs if self.eligible(r)]
        if reqs:
            self._round(reqs)
            if ENV.MTP_CHAIN:
                # Chain straight into the next round on the just-published bonus instead of
                # letting a plain decode step run in between (that step would spend a full
                # decode for a single token and roughly halve the speculative benefit).
                # The next iteration's run_round resets _deferred, so this only suppresses
                # the decode batch of THIS iteration.
                self._deferred = set(reqs)

    def _accept(self, reqs, logits, am, draft_mat, k, dev):
        """Per-request acceptance over the verify logits (flat row ``i*(k+1)+j`` predicts
        the token at ``C+j+1``).

        Greedy: the target argmax at verify position ``C+j`` must equal draft ``d_j``;
        reject at the first mismatch, bonus = target argmax at the last accepted position.
        Sampled: vLLM-style speculative rejection sampling against the target's truncated
        distribution -- the greedy draft is a point-mass proposal, so accept ``d`` with
        probability ``p_target(d)`` and on rejection draw from ``normalize(p - delta_d)``.
        Mirrors ``DSparkManager._accept``.
        """
        a_list: List[int] = []
        bonus_list: List[int] = []
        for i, r in enumerate(reqs):
            sp = r.sampling_params
            if sp.is_greedy:
                nomatch = am[i, :k] != draft_mat[i]
                a = int(nomatch.to(torch.int64).argmax()) if bool(nomatch.any()) else k
                bonus = int(am[i, a])
            else:
                probs = probs_from_logits(logits[i * (k + 1) : (i + 1) * (k + 1)], sp)
                a, bonus = k, -1
                for j in range(k):
                    d = int(draft_mat[i, j])
                    if float(torch.rand(1, device=dev)) < float(probs[j, d]):
                        continue
                    a = j
                    bonus = int(sample_residual(probs[j], d))
                    break
                if bonus < 0:
                    bonus = int(torch.multinomial(probs[k], 1))
            a_list.append(a)
            bonus_list.append(bonus)
        return a_list, bonus_list

    def _round(self, reqs: List[Req]) -> None:
        """One synchronous speculative round (normal_loop): draft+verify, then accept+
        publish+commit. ``overlap_loop`` drives the two halves separately so the previous
        batch drains on the host while the verify runs."""
        self.finish_round(self.begin_round(reqs))

    def _timing_lap(self):
        """The FREETOKEN_MTP_TIMING lap closure (a no-op when disabled). Each half of a
        round gets a fresh clock; the per-name accumulators are shared."""
        if not self._timing_on:
            return lambda name: None
        import time as _time

        _tt = _time.perf_counter()

        def _lap(name: str) -> None:
            nonlocal _tt
            torch.cuda.synchronize()
            now = _time.perf_counter()
            self._timing_acc[name] = self._timing_acc.get(name, 0.0) + (now - _tt)
            _tt = now

        return _lap

    def begin_round(self, reqs: List[Req]) -> _RoundState:
        """Draft/verify half of a round: allocate the verify window, run the draft chain,
        snapshot the GDN state and LAUNCH the verify. On return the host ``device_len``/
        ``input_ids`` are restored to the round-entry geometry, so an interleaved drain of
        the previous batch appends its pending token at ``cached_len``; the verify's own
        effects live in GPU buffers (and ``finish_round`` re-points the host view forward)."""
        sched, eng = self.sched, self.engine
        cm = sched.cache_manager
        pool = eng.linear_state_pool
        k, n = self.k, len(reqs)
        dev = eng.device
        C = [r.cached_len for r in reqs]
        assert all(r.device_len == c + 1 for r, c in zip(reqs, C)), (
            "MTP round entry expects the pending-token invariant cached_len == device_len - 1"
        )
        _lap = self._timing_lap()
        _moe = getattr(eng, "moe_offload_cache", None)
        if self._timing_on and _moe is not None:
            _moe.collect_stats = True
            _moe.reset_stats()

        # -- 1. grow the host ids with placeholders; allocate the verify pages [C, C+k+1)
        for r in reqs:
            r._ids_buf[r.device_len : r.device_len + k] = 0
            r.input_ids = r._ids_buf[: r.device_len + k]
            r.device_len += k
        verify_batch = Batch(reqs=reqs, phase="prefill")
        sched._prepare_batch(verify_batch)
        from .scheduler import _make_input_tuple  # lazy: mtp wird von scheduler importiert

        inp_map, inp_pos = _make_input_tuple(verify_batch, dev)
        # (mapping, positions) as ONE advanced-index pair: indexing with the mapping
        # alone would slice the token_pool's first axis and hand the model a 2-D [T,
        # max_len] block instead of the per-position gather. The actual gather is
        # deferred to just before the verify forward: the draft ids are only staged
        # into token_pool in step 3, and extend_forward reads batch.input_ids as-is
        # (DSpark stages before its gather for the same reason).
        _lap("prep")

        # -- 2. draft chain: k sequential single-token MTP forwards. The anchor token is
        #       the pending token staged in token_pool at C (in overlap the host append
        #       only happens in the interleaved drain), so read it on-device.
        tables_t = torch.tensor([r.table_idx for r in reqs], dtype=torch.int64, device=dev)
        toks = sched.token_pool[
            tables_t, torch.tensor(C, dtype=torch.int64, device=dev)
        ].to(torch.int32)
        R_last = torch.stack([r.mtp_streams_row for r in reqs], dim=0)
        drafts_gpu: List[torch.Tensor] = []
        dgr = getattr(eng, "draft_graph_runner", None)
        if dgr is not None and dgr.can_use(n, max(r.device_len for r in reqs)):
            # Phase 3b: replay the captured draft step k times, staging the per-step
            # positions/out_loc/MTP-KV length in between (the graph bakes the kernels, not
            # the geometry). The R_next/token hand-off stays eager (a copy and an argmax).
            tables = [r.table_idx for r in reqs]
            for j in range(k):
                pos = torch.tensor([c + j for c in C], dtype=torch.int32, device=dev)
                out_loc = eng.page_table[
                    torch.tensor(tables, dtype=torch.int64, device=dev), pos.to(torch.int64)
                ]
                R_next, logits = dgr.replay_step(
                    tables, [c + j + 1 for c in C], pos, out_loc.to(torch.int32), R_last, toks
                )
                d = torch.argmax(logits, dim=-1).to(torch.int32)
                drafts_gpu.append(d)
                toks = d
                R_last = R_next
        else:
            for j in range(k):
                pos = torch.tensor([c + j for c in C], dtype=torch.int32, device=dev)
                sb = self._synth_batch(
                    [_DraftReq(r.table_idx, c + j, c + j + 1) for r, c in zip(reqs, C)],
                    pos,
                    eng.page_table[tables_t, pos.to(torch.int64)],
                )
                with eng.ctx.forward_batch(sb):
                    R_next, hidden = self.mtp.draft_step(self.embed, R_last, toks, sb)
                    logits = self.lm_head.forward(hidden)  # 1 token/req: last-row select = the row
                d = torch.argmax(logits, dim=-1).to(torch.int32)
                drafts_gpu.append(d)
                toks = d
                R_last = R_next
        _lap("draft")

        # -- 2b. n-gram combiner: a repetition found in the request's own history is a
        #        stronger draft than the MTP head. Override the chain (all positions the
        #        match covers) before staging; the verify and acceptance are unchanged.
        if self.ngram_enabled:
            for i, r in enumerate(reqs):
                ng = ngram_draft(r.input_ids.tolist(), self.ngram_size, k)
                if not ng:
                    continue
                self.stats_ngram_rounds += 1
                for j in range(min(k, len(ng))):
                    drafts_gpu[j][i] = ng[j]

        # -- 3. stage the draft ids into token_pool (the verify gather reads them on-device).
        #       The host ids feed stop-string checks and the radix insert and are staged in
        #       finish_round, so this half stays free of host syncs.
        for j in range(k):
            for i, r in enumerate(reqs):
                sched.token_pool[r.table_idx, C[i] + 1 + j] = drafts_gpu[j][i]
        _lap("stage")

        # -- 4. snapshot the GDN live state; the verify over-advances it in place
        hybrid = pool is not None and cm.is_hybrid
        scratch: List[int] = []
        if pool is not None:
            if hybrid:
                cm.ensure_mamba_slots(n)
                scratch = pool.alloc(n)
            for i, r in enumerate(reqs):
                live = (
                    r.linear_slot_idx
                    if (hybrid and r.linear_slot_idx is not None)
                    else r.table_idx
                )
                if hybrid:
                    pool.copy_from(live, scratch[i])
        _lap("snapshot")

        # -- 5. verify forward (k+1 tokens per request, full per-position logits). Gather
        #       NOW, after step 3 staged the drafts into token_pool: the verify must be
        #       conditioned on the pending token + the actual draft chain, not placeholders.
        verify_batch.input_ids = sched.token_pool[inp_map, inp_pos]
        # Flag the batch so the GDN layers capture their kernel inputs for the
        # accepted-prefix commit (Phase 2).
        verify_batch.mtp_verify = True
        vgr = getattr(eng, "verify_graph_runner", None)
        if vgr is not None and vgr.can_use(n, max(r.device_len for r in reqs)):
            # Phase 3: replay the captured verify extend (fixed n_max x (k+1) shape).
            logits, verify_streams = vgr.replay(verify_batch)
        else:
            logits, verify_streams = eng.extend_forward(verify_batch)
        _lap("verify")

        # Restore the host geometry to the round entry so an interleaved drain appends the
        # previous batch's pending token at cached_len; finish_round re-points forward.
        for r, c in zip(reqs, C):
            r.device_len = c + 1
            r.input_ids = r._ids_buf[:c]

        return _RoundState(
            reqs, C, k, n, dev, verify_batch, logits, verify_streams, drafts_gpu,
            toks, hybrid, scratch, pool, _moe, self._timing_on,
        )

    def finish_round(
        self, state: _RoundState, deferred_finish: Optional[Set[Req]] = None
    ) -> None:
        """Accept/publish half of a round. ``deferred_finish`` holds requests whose
        terminal token was already published by an interleaved drain (overlap): they are
        not published again, but their verify tail is rolled back and their resources are
        freed here (the drain deferred the free so the in-flight verify stayed safe)."""
        sched, eng = self.sched, self.engine
        cm = sched.cache_manager
        reqs, C, k, n, dev = state.reqs, state.C, state.k, state.n, state.dev
        pool, scratch, hybrid = state.pool, state.scratch, state.hybrid
        logits, verify_streams, drafts_gpu = state.logits, state.verify_streams, state.drafts_gpu
        _moe = state.moe
        deferred = deferred_finish or set()
        _lap = self._timing_lap()

        drafts_host = [d.tolist() for d in drafts_gpu]
        for j in range(k):
            for i, r in enumerate(reqs):
                r._ids_buf[C[i] + 1 + j] = drafts_host[j][i]
        am = torch.argmax(logits, dim=-1).view(n, k + 1)
        draft_mat = torch.stack(drafts_gpu, dim=1).to(torch.int64)
        a_list, bonus_list = self._accept(reqs, logits, am, draft_mat, k, dev)
        bonus_gpu = torch.tensor(bonus_list, dtype=torch.int32, device=dev)
        # cumulative acceptance accounting (published via the reply stamps)
        self.stats_rounds += n
        self.stats_drafted += n * k
        self.stats_accepted += sum(a_list)
        for _a in a_list:
            self.a_hist[_a] += 1
        # Per-draft-index argmax-match counts kept ON DEVICE (no per-round host sync): a
        # draft-quality proxy (does draft 0 match, how fast does it decay?). Read in the log.
        self.stats_match += (am[:, :k] == draft_mat).sum(dim=0)
        _lap("accept")

        # -- 6. publish accepted drafts + bonus; roll back the rejected pages
        reply: List[DetokenizeMsg] = []
        plans: List[Tuple[Req, int, int, bool, bool]] = []  # (req, rext_end, keep_to, finished, bonus_published)
        for i, r in enumerate(reqs):
            if r in deferred:
                # The interleaved drain published this request's terminal token and removed
                # it from decode_manager, but deferred the free so the in-flight verify was
                # safe. Roll the verify tail back to the committed terminal token; the
                # commit/free loop below releases it exactly once.
                r.device_len = C[i] + k + 1
                cm.rollback_last(r, C[i] + 1)
                r.cached_len = C[i] + 1
                r.input_ids = r._ids_buf[: C[i] + 1]
                plans.append((r, C[i] + 1, C[i] + 1, True, False))
                continue
            a = a_list[i]
            bonus = bonus_list[i]
            m = 0  # accepted drafts published
            finished = False
            finish_reason = None
            matched_stop = None
            # Rebuild the host ids from the pending token forward: step 1 grew input_ids to
            # C+1+k, so append_host below would write the published tokens PAST the accepted
            # prefix (and toolcall_anchor_len / the length guards would read the wrong numel).
            # Re-point to the committed prefix so appends land at C+1, C+2, ... exactly as the
            # drain does; rejected drafts beyond the accepted prefix are dropped by the later
            # truncation.
            r.input_ids = r._ids_buf[: C[i] + 1]

            def _check(tok: int) -> Tuple[bool, str | None, str | None]:
                hit_length = r.output_budget_exhausted
                hit_eos = (
                    not r.sampling_params.ignore_eos and tok in sched.eos_token_ids
                )
                ms = (
                    sched._match_stop_str(r)
                    if (not hit_eos and r.sampling_params.stop_strs)
                    else None
                )
                fin = hit_length or hit_eos or ms is not None
                return fin, ("stop" if (hit_eos or ms is not None) else "length") if fin else None, ms

            for j in range(a):
                if r.output_budget_exhausted:
                    # The committed prefix is already at max_device_len (a prior round
                    # filled it): stop before the append that would overflow _ids_buf.
                    finished, finish_reason = True, "length"
                    break
                tok = drafts_host[j][i]
                r.append_host(torch.tensor([tok]))
                m += 1
                finished, finish_reason, matched_stop = _check(tok)
                if (
                    tok == sched.toolcall_anchor_id
                    and r.toolcall_anchor_len is None
                    and not finished
                ):
                    r.toolcall_anchor_len = r.input_ids.numel()
                reply.append(
                    DetokenizeMsg(
                        uid=r.uid, next_token=tok, finished=finished,
                        finish_reason=finish_reason, matched_stop=matched_stop,
                        stop_strs=r.sampling_params.stop_strs or None,
                    )
                )
                if finished:
                    break
            bonus_published = not finished
            if bonus_published and not r.output_budget_exhausted:
                r.append_host(torch.tensor([bonus]))
                finished, finish_reason, matched_stop = _check(bonus)
                if (
                    bonus == sched.toolcall_anchor_id
                    and r.toolcall_anchor_len is None
                    and not finished
                ):
                    r.toolcall_anchor_len = r.input_ids.numel()
                reply.append(
                    DetokenizeMsg(
                        uid=r.uid, next_token=bonus, finished=finished,
                        finish_reason=finish_reason, matched_stop=matched_stop,
                        stop_strs=r.sampling_params.stop_strs or None,
                    )
                )
            finished_at_draft = finished and not bonus_published
            # re-extend range: pending token + accepted drafts (NOT the bonus)
            rext_end = C[i] + 1 + (m if finished_at_draft else a)
            keep_to = rext_end + (0 if finished_at_draft else 1)
            # begin_round restored device_len to the round entry; rollback_last bounds the
            # freed tail by device_len, so restore the verify's grown end first.
            r.device_len = C[i] + k + 1
            # Roll back to the COMMITTED prefix, not to keep_to: the pending bonus token's page
            # is allocated by the NEXT step's allocate_paged over [cached_len, device_len). If
            # rext_end is page-aligned and a draft was rejected (a < k), the verify allocated a
            # page beyond page_ceil(rext_end); rolling back only to keep_to would retain it, and
            # the next allocate_paged (first_page = div_ceil(cached_len, ps)) would re-allocate
            # that page index and orphan the retained page -> "integrity check failed" at idle.
            cm.rollback_last(r, rext_end)
            r.device_len = keep_to
            r.input_ids = r._ids_buf[:keep_to]
            if bonus_published:
                sched.token_pool[r.table_idx, rext_end] = bonus_gpu[i]
            plans.append((r, rext_end, keep_to, finished, bonus_published))
        _lap("publish")

        # -- 7. restore the pre-verify state, then commit the accepted prefix (Phase 2:
        #       no re-extend). The chunk verify over-advanced the live state; restore it
        #       from the snapshot, then commit_verify replays exactly the accepted tokens
        #       into the live conv + SSM slots from the captured inputs.
        if pool is not None and hybrid:
            live_slots = []
            for i, (r, _rext, _keep, _fin, _bp) in enumerate(plans):
                live = (
                    r.linear_slot_idx
                    if (hybrid and r.linear_slot_idx is not None)
                    else r.table_idx
                )
                pool.copy_from(scratch[i], live)
                live_slots.append(live)
            lens = [p[1] - C[i] for i, p in enumerate(plans)]
            # n=1 uses the captured commit graph (kills the per-token launch tail); n>1 and
            # any shape the graph lacks fall back to the eager per-token commit.
            cgr = getattr(eng, "commit_graph_runner", None)
            if cgr is not None and cgr.can_use(n, lens[0]):
                cgr.replay(live_slots[0], lens[0])
            else:
                eng.model.commit_mtp_verify(pool, lens, live_slots)
            pool.free(scratch)
        for i, (r, rext_end, keep_to, finished, bonus_published) in enumerate(plans):
            r.cached_len = rext_end
            if bonus_published and not finished:
                # the bonus is staged but NOT processed: next round's pending token
                r.device_len = keep_to
                r.input_ids = r._ids_buf[:keep_to]
            if verify_streams is not None:
                # residual at the last committed position C+a (= rext_end - 1), the next
                # draft chain's R_last.
                r.mtp_streams_row = verify_streams[i * (k + 1) + (rext_end - C[i]) - 1].clone()
            if finished:
                sched.decode_manager.remove_req(r)
                sched._free_req_resources(r)
                sched.finished_reqs.add(r)
        _lap("commit")

        # Diagnostics for the qwen4_exp QSA/hybrid page-accounting crash
        # (`CacheManager integrity check failed`): per-request round geometry + the
        # live free-page count. Off unless FREETOKEN_MTP_DEBUG=1.
        if os.environ.get("FREETOKEN_MTP_DEBUG"):
            for i, (r, rext_end, keep_to, finished, bp) in enumerate(plans):
                logger.info_rank0(
                    f"MTP debug req{i}: table={r.table_idx} C={C[i]} k={k} a={a_list[i]} "
                    f"rext_end={rext_end} keep_to={keep_to} fin={finished} bonus={bp} "
                    f"dev_len={r.device_len} cached={r.cached_len} free_slots={len(cm.free_slots)}"
                )
            logger.info_rank0(
                f"MTP debug round: uids={[r.uid for r in reqs]} C={C} "
                f"anchor={state.anchor_toks.tolist()} "
                f"a={a_list} bonus={bonus_list} drafts={drafts_host} "
                f"reply={[(m.next_token, m.finished) for m in reply]}"
            )

        # -- 8. publish
        if reply:
            used, total = sched._kv_usage_pages()
            mamba_slots = sched._mamba_slot_usage()
            swa_tokens = sched._swa_token_usage()
            mem = sched._gpu_mem_bytes()
            mamba_used, mamba_total = mamba_slots or (0, 0)
            swa_used, swa_total = swa_tokens or (0, 0)
            for msg in reply:
                msg.kv_used_pages = used
                msg.kv_total_pages = total
                msg.mamba_used_slots = mamba_used
                msg.mamba_total_slots = mamba_total
                msg.swa_used_tokens = swa_used
                msg.swa_total_tokens = swa_total
                msg.gpu_mem_bytes = mem
                msg.mtp_enabled = True
                msg.mtp_drafted = self.stats_drafted
                msg.mtp_accepted = self.stats_accepted
            self.stats_rounds_logged += n
            if self.stats_rounds_logged >= 100:
                self.stats_rounds_logged = 0
                seen = max(1, self.stats_rounds)  # cumulative rounds = one draft per index
                curve = " ".join(
                    f"d{j}={int(h) / seen:.2f}" for j, h in enumerate(self.stats_match.tolist())
                )
                ran = sum(self.a_hist)
                cond = " ".join(
                    f"a>={j}:{sum(self.a_hist[j:]) / max(1, ran):.2f}"
                    for j in range(self.k + 1)
                )
                logger.info_rank0(
                    f"MTP: drafted {self.stats_drafted}, accepted {self.stats_accepted} "
                    f"(rate {self.stats_accepted / max(1, self.stats_drafted):.2f}) "
                    f"| per-draft {curve} | {cond} | ngram_rounds={self.stats_ngram_rounds}"
                )
            sched.send_result(reply)
        _lap("send")

        if state.timing:
            self._timing_n += 1
            if self._timing_n % 100 == 0:
                avg = " ".join(
                    f"{name}={1000 * v / self._timing_n:.1f}ms"
                    for name, v in self._timing_acc.items()
                )
                moe_stats = ""
                if _moe is not None:
                    s = _moe.decode_miss_stats()
                    tot_missing = s["missing_per_layer"] * s["layer_calls"]
                    moe_stats = (
                        f" | verify-MoE calls={s['layer_calls']} "
                        f"miss/layer={s['missing_per_layer']:.1f} "
                        f"active/layer={s['active_per_layer']:.1f} "
                        f"miss_rate={s['miss_rate']:.2f} "
                        f"missing_total={tot_missing:.0f}"
                    )
                tok = sum(p[1] - C[i] for i, p in enumerate(plans))
                logger.info_rank0(
                    f"MTP timing (avg over 100 rounds): {avg}{moe_stats} | tokens/round={tok}"
                )
                self._timing_acc.clear()
