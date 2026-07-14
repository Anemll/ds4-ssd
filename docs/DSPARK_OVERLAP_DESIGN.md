# DSpark draft/verify overlap — design (from recon 2026-07-06)

Goal: hide ~10.5ms/block draft time. Serial g_queue (ds4_metal.m:54/5132) => win comes from
removing the CPU round-trip, not GPU concurrency.

## Facts
- Draft trunk (metal_graph_eval_dspark_draft dspark.c:3261; prefetch variant :3019) writes the
  SAME batch_*/spec_logits bank the verifier uses (full table in the recon; verifier:
  metal_graph_verify_decodeN_attn_exact_ffn_batch dspark.c:5147).
- Post-verify the verifier still needs: g->spec_logits (read ds4.c:31869 via :22724, dspark.c:5894),
  batch_cur_hc/batch_next_hc (commit read ds4.c:31807/31876 via :22787), comp_selected
  (dspark.c:5860/5877), dspark_kv_cache (commit update dspark.c:1693 @ds4.c:31809/31877).
- CPU blocks on verify at dspark.c:5667 (main end_commands) and :5828 (head cb end).
- VERIFY_SPLIT_LAYERS=4 default-on (dspark.c:140), flush non-blocking (:5369).
- Prefetch prior art: start dspark.c:3019 (submit_commands :3071 non-blocking), finish :3087
  (synchronize :3100); session wrapper ds4.c:26393, seed argmax(s->logits) :26421; call sites
  post-commit: ds4.c:31851/31916/31964, 32632/32698/32750/32994/33070; misspec guard :26355-26360.
- ensure_prefill_scratch_rows may resize the bank mid-flight (ssd/ssd_flash_moe_allocation.c:147-165).

## Stage A1-A3 (byte-safe, land first)
A1 struct: add private draft bank to ds4_gpu_graph (ds4.c:9230-9280): dspark_draft_ mirrors of:
  cur_hc,next_hc,flat_hc,hc_mix,hc_split,attn_cur,attn_norm,qr,qr_norm,q,kv_raw,kv,heads,attn_low,
  attn_out,after_attn_hc,ffn_cur,ffn_norm,shared_{gate,up,mid,out},router_{logits,probs,selected,
  weights},routed_{gate,up,mid,out},low_tmp,spec_logits,input_ids,h. Free with the others (~ds4.c:9977).
A2 allocator: clone metal_graph_ensure_prefill_scratch_rows (ssd_flash_moe_allocation.c:138-292)
  as metal_graph_ensure_dspark_draft_scratch_rows with pc=draft_cap(<=6); spec_logits mirror =
  draft_cap x DS4_N_VOCAB x 4. Call at top of prefetch_start (replaces dspark.c:3042 call there).
A3 routing: in metal_graph_eval_dspark_draft_prefetch_start ONLY (dspark.c:3056-3072): save all
  g->batch_*/spec_logits/dspark_input_ids/dspark_h pointers, repoint to mirrors, run
  seed_block/three_layer_forward/markov_chain_fast_encode, restore pointers (same idiom as
  verifier HC save/restore dspark.c:5210-5211/5682-5683). prefetch_finish (:3087) must read
  the MIRROR dspark_draft_input_ids (:3103) — keep pointers consistent there too.
  Launch sites unchanged (post-commit). Byte-safe by construction: live draft path untouched.

## Stage A4 (full win, later): move launch pre-readback: head cb ends submit_commands (:5828->
  submit), GPU argmax of spec_logits row draft_n-1 seeds mirror input_ids, speculative
  update_main_kv_range before draft trunk, single synchronize drains verify+draft, memcpy
  readbacks after; misspec net = existing finish guard. HAZARD list in recon (spec_logits/HC/
  comp_selected protected by mirrors; KV is the remaining speculated dependency).

## Validation for A1-A3
1. make ds4 clean. 2. cmp gate: default run vs DS4_DSPARK_DRAFT_PREFETCH=1 run, same prompt
  (space-invaders, -n 2000): outputs byte-identical AND both identical to pre-change default.
3. perf pair (n=3000): prefetch on vs off; expect small + (draft partially hidden behind
  commit ops); no regression allowed. Log to DSPARK_PLUS10_LOOP.md.
