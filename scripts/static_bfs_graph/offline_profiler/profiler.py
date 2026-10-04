"""Exact first-order edge observations from shared source trunks and verifier forks."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
import math
from pathlib import Path
import time

import numpy as np
import torch

from scripts.static_bfs_graph.graph_builder import all_edges, dense_reference, edge_intervention
from scripts.static_bfs_graph.runner import write_atomic

from modal_n8_gpu.prefix_replay import STEPS, _tensor_sha
from .replay import denoise_step, restore
from .snapshot import ReplaySnapshot
from .source_trunk import build_source_trunk
from .verifier_fork import apply_cached_verifier, shared_verifier

TAUS = (2.0, 8.0, 32.0)


def _terminal(runtime, latents, generator, prompt: str):
    pipe = runtime.pipe
    latent_hash = _tensor_sha(latents)
    tensor = pipe.vae.decode(latents / pipe.vae.config.scaling_factor,
                             return_dict=False, generator=generator)[0]
    images = list(pipe.image_processor.postprocess(tensor, output_type="pil",
                                                   do_denormalize=[True] * 8))
    torch.cuda.synchronize()
    scores = [float(x) for x in runtime.reward.score_batched([prompt] * 8, images)]
    selected = int(np.argmax(np.asarray(scores, dtype=np.float64)))
    rgb = np.asarray(images[selected].convert("RGB"), dtype=np.uint8)
    if len(scores) != 8 or not all(math.isfinite(x) for x in scores) or rgb.std() <= 1:
        raise RuntimeError("invalid terminal reward")
    return {"final_latent_sha256": latent_hash, "final_particle_scores": scores,
            "selected_particle_index": selected, "selected_final_score": scores[selected],
            "selected_image_rgb_sha256": sha256(rgb.tobytes()).hexdigest(),
            "selected_image_rgb_std": float(rgb.std())}


@torch.no_grad()
def _branch_suffix(runtime, pre_event: ReplaySnapshot, policy, cached, prompt_index: int,
                   trial_seed: int, prompt: str):
    start = time.perf_counter()
    decoded, raw = cached
    latents, embeds, generator, adapter = apply_cached_verifier(
        runtime, pre_event, policy, decoded, raw, prompt_index, trial_seed)
    for i in range(pre_event.sampling_idx + 1, 100):
        latents, x0 = denoise_step(runtime.pipe, latents, embeds, generator, i)
        latents, _ = adapter.resample(sampling_idx=i, latents=latents, x0_preds=x0)
    terminal = _terminal(runtime, latents, generator, prompt)
    expected_steps = list(policy.resampling_steps)
    if [e["sampling_index"] for e in adapter.events] != expected_steps:
        raise RuntimeError("branch suffix event schedule changed")
    return {**terminal, "resampling_events": deepcopy(adapter.events),
            "branch_suffix_wall_seconds": time.perf_counter() - start,
            "branch_suffix_steps": 99 - pre_event.sampling_idx}


@torch.no_grad()
def profile_trajectory(runtime, checkpoint: dict, reference_row: dict,
                       output_dir: Path, only_edges: set[str] | None = None):
    prompt_index = checkpoint["prompt_index"]
    trial_seed = checkpoint["trial_seed"]
    prompt = checkpoint["prompt"]
    reference = dense_reference(STEPS, 8.0, 8)
    edges = all_edges(STEPS, TAUS)
    selected_edges = [e for e in edges if only_edges is None or e.key in only_edges]
    by_src = defaultdict(list)
    for edge in selected_edges:
        by_src[edge.src].append(edge)
    source_seconds = verifier_seconds = branch_seconds = 0.0
    source_steps = suffix_steps = shared_verifier_count = actual_branch_count = 0
    logical = 0
    results = {}
    for src, source_edges in by_src.items():
        snap = ReplaySnapshot.from_reference(checkpoint["snapshots"][src])
        destinations = tuple(sorted({edge.dst for edge in source_edges}))
        t0 = time.perf_counter()
        trunk, steps = build_source_trunk(runtime, snap, destinations,
                                          prompt_index, trial_seed)
        source_seconds += time.perf_counter() - t0
        source_steps += steps
        for dst in destinations:
            destination_edges = [e for e in source_edges if e.dst == dst]
            pre_event, x0_cpu = trunk[dst]
            if dst == 100:
                latents, _, generator, adapter = restore(
                    runtime, pre_event, reference, prompt_index, trial_seed)
                terminal = _terminal(runtime, latents, generator, prompt)
                for edge in destination_edges:
                    policy = edge_intervention(edge, reference)
                    if policy.resampling_steps != tuple(e["sampling_index"] for e in adapter.events):
                        raise RuntimeError("END edge schedule mismatch")
                    results[edge.key] = {**terminal, "resampling_events": deepcopy(adapter.events),
                                         "branch_suffix_wall_seconds": 0.0,
                                         "branch_suffix_steps": 0}
                    logical += 1
                continue
            t0 = time.perf_counter()
            example_policy = edge_intervention(destination_edges[0], reference)
            cached = shared_verifier(runtime, pre_event, x0_cpu, example_policy,
                                     prompt_index, trial_seed)
            verifier_seconds += time.perf_counter() - t0
            shared_verifier_count += 1
            for edge in destination_edges:
                policy = edge_intervention(edge, reference)
                row = _branch_suffix(runtime, pre_event, policy, cached,
                                     prompt_index, trial_seed, prompt)
                results[edge.key] = row
                branch_seconds += row["branch_suffix_wall_seconds"]
                suffix_steps += row["branch_suffix_steps"]
                actual_branch_count += 1
                logical += 1
        print("SHARED_SOURCE_DONE", {"src": src, "logical_edges": logical,
                                     "shared_verifier_evaluations": shared_verifier_count,
                                     "tau_suffix_branches": actual_branch_count}, flush=True)
        torch.cuda.empty_cache()
    if logical != len(selected_edges):
        raise RuntimeError("missing logical edge observations")
    output_dir.mkdir(parents=True, exist_ok=True)
    for edge in selected_edges:
        row = results[edge.key]
        policy = edge_intervention(edge, reference)
        record = {"backend": "CUDA_Modal", "stage": "shared_tree_edge",
                  "protocol": checkpoint["protocol"], "prompt_index": prompt_index,
                  "trial_seed": trial_seed, "edge_key": edge.key,
                  "edge": edge.record(), "policy_id": policy.id,
                  "reference_score": reference_row["selected_final_score"],
                  "delta": reference_row["selected_final_score"] - row["selected_final_score"],
                  "diffusion_NFE": 800, **row}
        write_atomic(output_dir / f"{edge.key}.json", record)
    return {"prompt_index": prompt_index, "trial_seed": trial_seed,
            "reference_passes": 1, "source_trunks": len(by_src),
            "source_trunk_seconds": source_seconds, "source_trunk_steps": source_steps,
            "shared_destination_verifier_evaluations": shared_verifier_count,
            "shared_verifier_seconds": verifier_seconds,
            "tau_suffix_branches": actual_branch_count,
            "tau_suffix_seconds": branch_seconds, "tau_suffix_steps": suffix_steps,
            "logical_edges": logical,
            "diffusion_steps_shared_profiler": 100 + source_steps + suffix_steps,
            "diffusion_steps_naive": 100 * logical,
            "verifier_evaluations_naive_at_destinations": actual_branch_count,
            "verifier_evaluations_shared_at_destinations": shared_verifier_count}
