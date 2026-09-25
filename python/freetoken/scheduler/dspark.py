"""DSpark speculative-decode rounds for DeepSeek-V4.1.

Mirrors the qwen4 ``MTPManager`` scheduler contract (``run_round`` before decode
scheduling, ``draft_prefill`` after a prefill forward), but the draft is ONE
semi-autoregressive block forward (``DSparkDraft.forward_spec``) that consumes the
target's aux hidden, and the verify is a prefill-phase extend at an ARBITRARY start
position (the V41Compressor supports it and captures carry snapshots so a rejected
tail can be rolled back to the accepted prefix).

Entry invariant per request: ``cached_len == device_len - 1`` with the freshly sampled
token pending at ``C = cached_len``. The draft predicts ``k = dspark_block_size`` tokens
``[C+1, C+k]`` from ``(pending@C, aux@[C-1])``; the verify runs ``[C, C+k]`` and greedy
accepts the matching prefix + bonus. Unlike MTP there is NO re-extend: the verify's KV
for the accepted positions is already correct and the compressor carry is restored from
the captured snapshot.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Tuple

import torch

from freetoken.core import Batch, Req
from freetoken.engine import DSPARK_VERIFY_MODES
from freetoken.engine.sample import probs_from_logits, sample_residual
from freetoken.message import DetokenizeMsg
from freetoken.utils import init_logger
from .dspark_controller import Decision, DSparkController, DSparkFaultLatch, FaultKind

if TYPE_CHECKING:
    from .scheduler import Scheduler

logger = init_logger(__name__)


def resolve_verify_mode(value: str) -> str:
    """Validate a DSpark verify mode (``--dspark-verify`` / ``FREETOKEN_DSPARK_VERIFY``)
    against the implemented set; empty resolves to the default."""
    mode = value or "prefill"
    if mode not in DSPARK_VERIFY_MODES:
        raise ValueError(
            f"dspark_verify must be one of {list(DSPARK_VERIFY_MODES)}, got {mode!r}"
        )
    return mode


def _flat(t: torch.Tensor) -> torch.Tensor:
    """[..., dim] -> a detached fp32 [tokens, dim] copy for the diagnostic diff."""
    return t.detach().reshape(-1, t.shape[-1]).float().clone()


class DSparkManager:
    uses_streams = False  # the draft reads the target aux buffer, not per-decode streams
    # one synchronous round (no begin/finish split): the scheduler keeps it on normal_loop
    overlaps = False

    def __init__(self, sched: "Scheduler") -> None:
        self.sched = sched
        self.engine = sched.engine
        model = sched.engine.model
        self.draft = model.dspark_draft() if hasattr(model, "dspark_draft") else None
        self.enabled = self.draft is not None
        # Env wins over the flag so a debug run can override the server config without
        # a restart (mirrors the other FREETOKEN_DSPARK_* knobs).
        self.verify_mode = resolve_verify_mode(
            os.environ.get("FREETOKEN_DSPARK_VERIFY")
            or getattr(sched.config, "dspark_verify", "prefill")
        )
        self._deferred: set[Req] = set()
        # per table row: the target-aux window [w, d*L] for positions [start, start+w)
        self._aux: dict[int, torch.Tensor] = {}
        self._aux_start: dict[int, int] = {}
        self._draft_logits: dict[int, torch.Tensor] = {}
        self._draft_conf: dict[int, torch.Tensor] = {}
        # per request (table row): the admission controller + the fault latch
        self._ctl: dict[int, DSparkController] = {}
        self._latch: dict[int, DSparkFaultLatch] = {}
        # table rows advanced by the last spec round (so a plain decode step can price
        # only the requests that REALLY decoded serially)
        self._spec_iter: set[int] = set()
        self.stats_drafted = 0
        self.stats_accepted = 0
        self.stats_rounds = 0
        self.stats_rounds_logged = 0
        if not self.enabled:
            self.k = 0
            return
        self.k = int(os.environ.get("FREETOKEN_DSPARK_K", self.draft.block_size))
        args = self.draft.args
        self.width = int(args.hidden_size) * len(args.dspark_target_layer_ids)
        logger.info_rank0(
            f"DSpark speculative decoding enabled: block k={self.k}, verify={self.verify_mode}, "
            "greedy + rejection sampling"
        )

    # --------------------------------------------------------------- eligibility
    def eligible(self, req: Req) -> bool:
        latch = self._latch.get(req.table_idx)
        return (
            self.enabled
            and req.table_idx != -1
            and (latch is None or not latch.drafter_disabled())
            and req.table_idx in self._aux
            and not req.aborted
            and req.mm_embeds is None
            and req.device_len == req.cached_len + 1
            and req.remain_len >= self.k + 1
        )

    # ------------------------------------------------------- prefill-side hook
    def draft_prefill(self, batch: Batch, pre_lens: List[Tuple[int, int]]) -> None:
        """After a normal prefill forward: seed each request's DSpark window from the
        just-forwarded target aux and remember the last position for the first draft."""
        if not self.enabled or not batch.is_prefill:
            return
        aux = self.engine.model.get_dspark_aux_hidden()
        if aux is None:
            return
        off = 0
        for r, (s, e) in zip(batch.reqs, pre_lens):
            n = e - s
            if n <= 0:
                continue
            seg = aux[off : off + n]
            off += n
            self.draft.seed_window(seg.unsqueeze(0), s)
            self._aux[r.table_idx] = seg[-1:].clone()
            self._aux_start[r.table_idx] = e - 1

    # ----------------------------------------------------------- the round
    def run_round(self) -> None:
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
            if os.environ.get("FREETOKEN_DSPARK_DIFF") == "1" and self.stats_rounds == 0:
                self._diff_round(reqs)
            else:
                self._round(reqs)

    def _draft_block(self, reqs: List[Req], C: List[int]) -> torch.Tensor:
        """One block draft per request: returns ``[n, k+1]`` = the anchor + k drafts."""
        if self.k == 0:
            return torch.tensor(
                [[int(r.input_ids[c])] for r, c in zip(reqs, C)],
                dtype=torch.int32, device=self.engine.device,
            )
        drafts: List[List[int]] = []
        for r, c in zip(reqs, C):
            anchor = int(r.input_ids[c])
            aux = self._aux[r.table_idx]
            start = self._aux_start[r.table_idx]
            ids = torch.tensor([anchor], dtype=torch.int64, device=self.engine.device)
            with self.engine.ctx.forward_batch(_DraftBatch()):
                out_ids, _logits, _conf = self.draft.forward_spec(ids, aux.unsqueeze(0), start)
            if _conf is not None:
                # the admission gate reads these every cycle, not only in debug
                self._draft_conf[r.table_idx] = _conf.detach()
            if os.environ.get("FREETOKEN_DSPARK_DEBUG") == "1" and _logits is not None:
                self._draft_logits[r.table_idx] = _logits.detach()
            # forward_spec always drafts ``block_size`` tokens; keep the first k so
            # FREETOKEN_DSPARK_K < block_size stays consistent with the verify length.
            drafts.append([int(t) for t in out_ids[0][: self.k + 1].tolist()])
        return torch.tensor(drafts, dtype=torch.int32, device=self.engine.device)

    def _controller(self, r: Req) -> DSparkController:
        ctl = self._ctl.get(r.table_idx)
        if ctl is None:
            ctl = DSparkController.from_env(block=self.draft.block_size)
            ctl.start_request()
            self._ctl[r.table_idx] = ctl
        return ctl

    def note_serial(self, wall_ms: float, reqs: List[Req]) -> None:
        """A plain decode step ran for ``reqs``: feed each request's controller the serial
        cost (DwarfStar's windowed cost feedback). Requests advanced by this iteration's
        spec round are skipped — they did not decode serially."""
        for r in reqs:
            if r.table_idx == -1 or r.table_idx in self._spec_iter:
                continue
            self._controller(r).note_serial(wall_ms, consumed=1)

    def _admit(self, reqs: List[Req], C: List[int], draft_mat: torch.Tensor):
        """Confidence admission (DwarfStar DSPARK-V41 §4): verify only the requests whose
        drafted prefix is confident enough. A declined cycle is a chosen call (it takes a
        window slot, its net is a straight loss); a request in the entry wait or a
        cooldown contributes nothing and falls through to serial."""
        keep: List[int] = []
        for i, r in enumerate(reqs):
            latch = self._latch.get(r.table_idx)
            if latch is not None and latch.drafter_disabled():
                continue
            ctl = self._controller(r)
            conf = self._draft_conf.get(r.table_idx)
            confs = conf[0][: self.k].tolist() if conf is not None else [1.0] * self.k
            decision, _admitted = ctl.decide(confs)
            if decision is Decision.ATTEMPT:
                keep.append(i)
            elif decision is Decision.DECLINE:
                # no drafter wall here: price the decline at one serial step (a loss)
                ctl.note_cycle(ctl.serial_ms() or 0.0, consumed=1, verified=False)
        if len(keep) == len(reqs):
            return reqs, C, draft_mat, len(reqs)
        if not keep:
            return [], [], draft_mat[:0], 0
        idx = torch.tensor(keep, dtype=torch.long, device=draft_mat.device)
        return (
            [reqs[i] for i in keep],
            [C[i] for i in keep],
            draft_mat.index_select(0, idx),
            len(keep),
        )

    def _note_drafter_fault(self, reqs: List[Req], exc: BaseException, *, undrained: bool) -> None:
        """Record a drafter fault on each affected request and skip its drafter. Drained
        (before the target verify) falls through to serial; undrained (after the batch,
        the target already consumed rows) additionally refuses the request."""
        kind = FaultKind.UNDRAINED if undrained else FaultKind.DRAINED
        for r in reqs:
            latch = self._latch.setdefault(r.table_idx, DSparkFaultLatch())
            latch.trip(kind, type(exc).__name__)
            self._ctl.pop(r.table_idx, None)
            if undrained:
                r.aborted = True
        logger.warning_rank0(
            "DSpark drafter fault (%s) — %s: %r"
            % (kind.value, "request refused" if undrained else "serial fallback", exc)
        )

    def _restore_carries(self, reqs: List[Req], capture: list, n_accepted: List[int],
                         C: List[int], k: int) -> None:
        """Roll each compressor register back to its request's accepted position.

        ``capture`` is the verify's ``(layer_id, tier, snapshot)`` list in call order:
        per (layer, segment) group, ``k+2`` snapshots (before token 0 + one per verify
        token). Restore index ``a+1`` (the state after the accepted token) and persist it
        to the ring block of that position's page, so the next decode/verify seeds from it.
        """
        backend = self.engine.attn_backend
        compressors = {(c.layer_id, c.tier): c for c in self.engine.model.dspark_compressors()}
        groups: dict[tuple, list] = {}
        for layer_id, tier, snap in capture:
            groups.setdefault((layer_id, tier), []).append(snap)
        seg_len = k + 2
        for (layer_id, tier), snaps in groups.items():
            comp = compressors.get((layer_id, tier))
            if comp is None:
                continue
            for i, (r, a, c) in enumerate(zip(reqs, n_accepted, C)):
                snap = snaps[i * seg_len + a + 1]
                slot = int(backend.window_slots_of(r.table_idx, c + a, c + a + 1).item())
                comp.restore_carry(snap, slot)

    # --------------------------------------- differential harness (FREETOKEN_DSPARK_DIFF)
    def _install_diff_wrappers(self, caps: dict) -> list:
        """Capture each target layer's attention input/output on both paths into ``caps``.

        The decode reference and the verify must go through the same ``Attention`` instances
        but different entry points, so the wrappers sit on the instances, not the class."""
        attns = []
        for layer in self.engine.model.model.layers.op_list:
            attn = layer.attn
            lid = attn.layer_id
            orig_pre = attn.forward_ragged
            orig_dec = attn.decode_step

            def pre(x, *args, _lid=lid, _o=orig_pre, **kw):
                out = _o(x, *args, **kw)
                caps.setdefault("verify", {})[_lid] = (_flat(x), _flat(out))
                return out

            def dec(x, *args, _lid=lid, _o=orig_dec, **kw):
                out = _o(x, *args, **kw)
                caps.setdefault("decode", {})[_lid] = (_flat(x), _flat(out))
                return out

            attn.forward_ragged = pre
            attn.decode_step = dec
            attns.append(attn)
        return attns

    def _decode_reference(self, reqs: List[Req], C: List[int]) -> torch.Tensor:
        """Eager plain-decode logits at position C per request: the reference the verify must
        reproduce. Builds a real decode-phase batch off the round-entry state (no graph)."""
        eng = self.engine
        dev = eng.device
        ref = Batch(reqs=list(reqs), phase="decode")
        eng.graph_runner.pad_batch(ref)
        B, n = ref.padded_size, len(reqs)
        pos = torch.zeros(B, dtype=torch.int32, device=dev)
        pos[:n] = torch.tensor(C, dtype=torch.int32, device=dev)
        ref.positions = pos
        ref.active_table_idx = torch.tensor(
            [r.table_idx for r in ref.padded_reqs], dtype=torch.int64, device=dev
        )
        ids = torch.zeros(B, dtype=torch.int32, device=dev)
        ids[:n] = torch.tensor(
            [int(r.input_ids[c]) for r, c in zip(reqs, C)], dtype=torch.int32, device=dev
        )
        ref.input_ids = ids.view(B, 1)
        eng.attn_backend.prepare_metadata(ref)
        with eng.ctx.forward_batch(ref), eng.model.forward_host_ctx(ref, False):
            logits = eng.model.forward()
        return logits[:n]

    def _diff_round(self, reqs: List[Req]) -> None:
        """Diagnostic (FREETOKEN_DSPARK_DIFF=1): one decode reference and one verify over the
        same position, diffed layer by layer, then stop. Localizes the greedy divergence on
        the real model instead of a separate harness."""
        sched, eng = self.sched, self.engine
        dev = eng.device
        k, n = self.k, len(reqs)
        C = [r.cached_len for r in reqs]
        caps: dict = {}
        attns = self._install_diff_wrappers(caps)
        try:
            ref_logits = self._decode_reference(reqs, C)
        except Exception as exc:  # noqa: BLE001 - diagnostic: report, still run the verify
            ref_logits = None
            logger.info_rank0(f"DSpark DIFF: decode reference failed: {type(exc).__name__}: {exc}")
        try:
            draft_mat = self._draft_block(reqs, C)
            for i, r in enumerate(reqs):
                r._ids_buf[r.cached_len + 1 : r.cached_len + 1 + k] = draft_mat[i, 1:].to(torch.int64)
                for j in range(k):
                    sched.token_pool[r.table_idx, C[i] + 1 + j] = draft_mat[i, j + 1]
                r.device_len = r.cached_len + k + 1
                r.input_ids = r._ids_buf[: r.device_len]
            verify_batch = Batch(reqs=reqs, phase="prefill")
            sched._prepare_batch(verify_batch)
            from .scheduler import _make_input_tuple

            inp_map, inp_pos = _make_input_tuple(verify_batch, dev)
            verify_batch.input_ids = sched.token_pool[inp_map, inp_pos]
            eng.ctx.dspark_carry_capture = []
            eng.ctx.dspark_decode_verify = self.verify_mode == "decode"
            try:
                verify_logits, _ = eng.extend_forward(verify_batch)
            finally:
                eng.ctx.dspark_carry_capture = None
                eng.ctx.dspark_decode_verify = False
        finally:
            for attn in attns:
                del attn.forward_ragged
                del attn.decode_step
        am = torch.argmax(verify_logits, dim=-1).view(n, k + 1)
        logger.info_rank0(f"DSpark DIFF C={C} k={k} drafts={draft_mat.tolist()}")
        if ref_logits is not None:
            logger.info_rank0(
                "DSpark DIFF logits max|decode-verify@C|="
                f"{(ref_logits.float() - verify_logits[:n].float()).abs().max().item():.3e} "
                f"argmax decode={torch.argmax(ref_logits, -1).tolist()} verify={am[:, 0].tolist()}"
            )
        for lid in sorted(caps.get("decode", {})):
            d_in, d_out = caps["decode"][lid]
            v_in, v_out = caps["verify"][lid]
            logger.info_rank0(
                f"DSpark DIFF layer={lid:>2} in max|d|={(d_in[0] - v_in[0]).abs().max().item():.3e} "
                f"out max|d|={(d_out[0] - v_out[0]).abs().max().item():.3e} "
                f"out_argmax_same={bool(torch.equal(d_out[0].argmax(-1), v_out[0].argmax(-1)))}"
            )
        raise RuntimeError("DSpark DIFF complete (diagnostic run)")

    def _accept(self, reqs: List[Req], logits: torch.Tensor, draft_mat: torch.Tensor, k: int):
        """vLLM-style acceptance over the verify logits.

        Greedy requests: the target argmax at verify position ``C+j`` must equal draft
        ``d_{j+1}`` (the draft is greedy, so the residual correction is that argmax). Sampling
        requests: speculative rejection sampling -- the greedy draft is a point-mass proposal,
        so accept ``d_{j+1}`` with probability ``p_target(d_{j+1})`` and on rejection draw from
        ``normalize(p - delta_d)``. Flat row ``i*(k+1)+j`` predicts the token at ``C+j+1``."""
        dev = self.engine.device
        a_list: List[int] = []
        bonus_list: List[int] = []
        for i, r in enumerate(reqs):
            row = logits[i * (k + 1) : (i + 1) * (k + 1)]
            sp = r.sampling_params
            if sp.is_greedy:
                am = torch.argmax(row, dim=-1)
                nomatch = am[:k] != draft_mat[i, 1:]
                a = int(nomatch.to(torch.int64).argmax()) if bool(nomatch.any()) else k
                bonus = int(am[a])
            else:
                probs = probs_from_logits(row, sp)
                a, bonus = k, -1
                for j in range(k):
                    d = int(draft_mat[i, j + 1])
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
        sched, eng = self.sched, self.engine
        cm = sched.cache_manager
        k, n = self.k, len(reqs)
        dev = eng.device
        C = [r.cached_len for r in reqs]
        import time as _time

        self._spec_iter = set()
        _timing = os.environ.get("FREETOKEN_DSPARK_TIMING") == "1"
        if _timing:
            torch.cuda.synchronize()
        _t0 = _time.perf_counter()

        # -- 1. draft block per request (one forward_spec each; k drafts + anchor)
        try:
            draft_mat = self._draft_block(reqs, C)  # [n, k+1] int32
        except Exception as exc:  # noqa: BLE001 — a drafter fault must not break serving
            self._note_drafter_fault(reqs, exc, undrained=False)
            return
        _t1 = _time.perf_counter()

        # 1b. confidence admission: verify only cycles the drafter is confident about;
        # declined / drafter-disabled requests fall back to the normal (serial) decode.
        reqs, C, draft_mat, n = self._admit(reqs, C, draft_mat)
        if n == 0:
            return
        # these requests are advanced by the spec round, not by a plain decode step
        self._spec_iter = {r.table_idx for r in reqs}

        # -- 2. grow the host ids with the verify tokens and allocate [C, C+k]
        for i, r in enumerate(reqs):
            r._ids_buf[r.cached_len + 1 : r.cached_len + 1 + k] = draft_mat[i, 1:].to(torch.int64)
            for j in range(k):
                sched.token_pool[r.table_idx, C[i] + 1 + j] = draft_mat[i, j + 1]
            r.device_len = r.cached_len + k + 1
            r.input_ids = r._ids_buf[: r.device_len]
        verify_batch = Batch(reqs=reqs, phase="prefill")
        sched._prepare_batch(verify_batch)
        from .scheduler import _make_input_tuple

        inp_map, inp_pos = _make_input_tuple(verify_batch, dev)
        verify_batch.input_ids = sched.token_pool[inp_map, inp_pos]

        # -- 3. verify forward (all-position logits) with carry capture
        eng.ctx.dspark_carry_capture = []
        eng.ctx.dspark_decode_verify = self.verify_mode == "decode"
        try:
            logits, _ = eng.extend_forward(verify_batch)
            capture = eng.ctx.dspark_carry_capture
        finally:
            eng.ctx.dspark_carry_capture = None
            eng.ctx.dspark_decode_verify = False
        if _timing:
            torch.cuda.synchronize()
        _t2 = _time.perf_counter()
        am = torch.argmax(logits, dim=-1).view(n, k + 1)

        # vLLM-style acceptance: greedy matches the target argmax, sampling uses speculative
        # rejection sampling (see _accept). The verify's own target is the reference.
        a_list, bonus_list = self._accept(reqs, logits, draft_mat, k)
        bonus_gpu = torch.tensor(bonus_list, dtype=torch.int32, device=dev)
        if os.environ.get("FREETOKEN_DSPARK_FORCE_A0") == "1":
            # Diagnostic: publish only the target's own token at C (a plain decode). If the
            # output then matches the no-spec reference, the verify logits are correct.
            a_list = [0] * n
            bonus_list = [int(torch.argmax(logits[i * (k + 1)], dim=-1)) for i in range(n)]
            bonus_gpu = torch.tensor(bonus_list, dtype=torch.int32, device=dev)
        if os.environ.get("FREETOKEN_DSPARK_DEBUG") == "1":
            ranks = []
            confs = []
            for i, r in enumerate(reqs):
                dl = self._draft_logits.get(r.table_idx)
                ranks.append(-1 if dl is None else int((dl[0, 0] > dl[0, 0][int(am[i, 0])]).sum()))
                c = self._draft_conf.get(r.table_idx)
                confs.append(None if c is None else round(float(c[0, 0]), 3))
            logger.info_rank0(
                f"DSpark round C={C} drafts={draft_mat.tolist()} am={am[:, :k].tolist()} "
                f"a={a_list} bonus={bonus_list} d1_rank={ranks} d1_conf={confs}"
            )
        self.stats_rounds += n
        self.stats_drafted += n * k
        self.stats_accepted += sum(a_list)

        # roll the compressor carry back to each accepted prefix
        try:
            self._restore_carries(reqs, capture, a_list, C, k)
        except Exception as exc:  # noqa: BLE001 — target consumed rows: refuse
            self._note_drafter_fault(reqs, exc, undrained=True)
            return

        # -- 4. publish accepted drafts + bonus
        aux = eng.model.get_dspark_aux_hidden()
        off = 0
        reply: List[DetokenizeMsg] = []
        for i, r in enumerate(reqs):
            a, bonus = a_list[i], bonus_list[i]
            seg = aux[off : off + k + 1]
            off += k + 1
            finished = False
            finish_reason = None
            matched_stop = None
            m = 0
            # Rebuild the host ids from the pending token forward: step 2 grew input_ids to
            # C+k+1, so append_host below would write the published tokens PAST the accepted
            # prefix (and the max_device_len guards would trip early). Re-point to the committed
            # prefix so appends land at C+1, C+2, ... exactly as the drain does.
            r.input_ids = r._ids_buf[: C[i] + 1]
            for j in range(a):
                if r.output_budget_exhausted:
                    finished, finish_reason = True, "length"
                    break
                tok = int(draft_mat[i, j + 1])
                r.append_host(torch.tensor([tok]))
                m += 1
                finished, finish_reason, matched_stop = _check(sched, r, tok)
                reply.append(_msg(r, tok, finished, finish_reason, matched_stop))
                if finished:
                    break
            bonus_published = not finished
            if bonus_published and not r.output_budget_exhausted:
                r.append_host(torch.tensor([bonus]))
                finished, finish_reason, matched_stop = _check(sched, r, bonus)
                reply.append(_msg(r, bonus, finished, finish_reason, matched_stop))
            # new state: accepted prefix [C, C+a] processed; bonus pending at C+a+1.
            # A finish DURING the drafts published only m tokens (MTP parity): committing
            # `a` here would advance cached_len/rollback past what was accepted and strand
            # the SWA slots of the over-counted positions (S14/S15 leak).
            finished_at_draft = finished and not bonus_published
            rext_end = C[i] + 1 + (m if finished_at_draft else a)
            keep_to = rext_end + (0 if finished_at_draft else 1)
            # Roll back to the COMMITTED prefix, not keep_to (same page-accounting rule as MTP):
            # the pending bonus token's page is allocated by the next allocate_paged over
            # [cached_len, device_len). Retaining a verify page beyond page_ceil(rext_end) when
            # rext_end is page-aligned would let the next allocate_paged orphan it (integrity leak).
            cm.rollback_last(r, rext_end)
            r.cached_len = rext_end
            r.device_len = keep_to
            r.input_ids = r._ids_buf[:keep_to]
            if os.environ.get("FREETOKEN_DSPARK_SWA_DEBUG") == "1" and cm.swa_paged:
                logger.info_rank0(
                    f"DSpark swa r={i} a={a} m={m} fin_draft={finished_at_draft} "
                    f"rext_end={rext_end} keep_to={keep_to} cached_len={r.cached_len} "
                    f"avail={cm.swa_pool.swa_available_size()}"
                )
            if bonus_published:
                sched.token_pool[r.table_idx, rext_end] = bonus_gpu[i]
            # next draft window: aux over the accepted positions [C, C+a]
            self._aux[r.table_idx] = seg[: a + 1].clone()
            self._aux_start[r.table_idx] = C[i]
            if finished:
                ctl = self._ctl.get(r.table_idx)
                if ctl is not None:
                    logger.info_rank0(
                        "DSpark request done: "
                        + " ".join(f"{key}={val}" for key, val in ctl.telemetry().items())
                    )
                sched.decode_manager.remove_req(r)
                sched._free_req_resources(r)
                sched.finished_reqs.add(r)
                self._aux.pop(r.table_idx, None)
                self._aux_start.pop(r.table_idx, None)
                self._ctl.pop(r.table_idx, None)
                self._latch.pop(r.table_idx, None)

        if reply:
            if os.environ.get("FREETOKEN_DSPARK_DEBUG") == "1":
                logger.info_rank0(
                    f"DSpark reply={[(m.next_token, m.finished, m.finish_reason) for m in reply]} "
                    f"eos={sorted(sched.eos_token_ids)}"
                )
            for msg in reply:
                msg.mtp_enabled = True
                msg.mtp_drafted = self.stats_drafted
                msg.mtp_accepted = self.stats_accepted
            sched.send_result(reply)
            self.stats_rounds_logged += n
            if self.stats_rounds_logged >= 100:
                self.stats_rounds_logged = 0
                logger.info_rank0(
                    f"DSpark: drafted {self.stats_drafted}, accepted {self.stats_accepted} "
                    f"(rate {self.stats_accepted / max(1, self.stats_drafted):.2f})"
                )
        if _timing:
            torch.cuda.synchronize()
            _t3 = _time.perf_counter()
            logger.info_rank0(
                f"DSpark TIMING n={n} k={k} draft={1e3 * (_t1 - _t0):.1f}ms "
                f"verify={1e3 * (_t2 - _t1):.1f}ms publish={1e3 * (_t3 - _t2):.1f}ms"
            )


class _DraftBatch:
    """Minimal batch view for a DSpark draft block: the draft only reads
    ``ctx.batch.is_prefill`` (its MoE dispatch); the draft attention is self-contained."""

    is_prefill = True
    is_decode = False


def _check(sched: "Scheduler", r: Req, tok: int):
    hit_length = r.output_budget_exhausted
    hit_eos = not r.sampling_params.ignore_eos and tok in sched.eos_token_ids
    ms = sched._match_stop_str(r) if (not hit_eos and r.sampling_params.stop_strs) else None
    fin = hit_length or hit_eos or ms is not None
    return fin, (("stop" if (hit_eos or ms is not None) else "length") if fin else None), ms


def _msg(r: Req, tok: int, finished: bool, finish_reason, matched_stop) -> DetokenizeMsg:
    return DetokenizeMsg(
        uid=r.uid, next_token=tok, finished=finished, finish_reason=finish_reason,
        matched_stop=matched_stop, stop_strs=r.sampling_params.stop_strs or None,
    )


__all__ = ["DSparkManager"]
