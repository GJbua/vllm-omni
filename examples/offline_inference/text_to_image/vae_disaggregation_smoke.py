# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""910B P0 checks: peer IPC, fixed-latent decode parity, and engine overlap.

Run from the repository root in the real Ascend environment. Both devices must
be visible to all processes. See vae_disaggregation_p0.md for commands.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing as mp
import os
import time
import traceback
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

import torch


def _latent(args):
    generator = torch.Generator(device="cpu").manual_seed(42)
    return torch.randn(1, (args.height // 16) * (args.width // 16), 64, generator=generator).to(
        getattr(torch, args.dtype)
    )


def _sender(args, tasks, copied, report):
    try:
        import torch_npu  # noqa: F401

        from vllm_omni.diffusion.worker.vae_ipc import export_npu_tensor, ipc_to_wire

        torch.npu.set_device(args.dit_device)
        tensor = _latent(args).to(f"npu:{args.dit_device}")
        payload = export_npu_tensor(tensor)
        tasks.put(json.dumps(ipc_to_wire(payload)))
        if not copied.wait(args.timeout):
            raise TimeoutError("VAE did not finish importing/copying the tensor.")
        # Keep the exporting process alive until peer use has finished.
        report.send({"role": "sender", "ok": True})
    except Exception:
        report.send({"role": "sender", "ok": False, "error": traceback.format_exc()})
    finally:
        report.close()


def _receiver(args, tasks, copied, report):
    try:
        import torch_npu  # noqa: F401

        from vllm_omni.diffusion.worker.vae_ipc import copy_npu_tensor, ipc_from_wire

        torch.npu.set_device(args.vae_device)
        device = torch.device("npu", args.vae_device)
        payload = ipc_from_wire(json.loads(tasks.get(timeout=args.timeout)))
        start = time.perf_counter()
        local = copy_npu_tensor(payload, device, getattr(torch, args.dtype))
        torch.testing.assert_close(local.cpu(), _latent(args), rtol=0, atol=0)
        copied.set()
        metrics = {"ipc_and_copy_s": time.perf_counter() - start}
        if args.mode == "decode":
            from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import QwenImagePipeline
            from vllm_omni.diffusion.worker.vae_worker import VAEDecodeTask, VAEWorkerProc

            config = SimpleNamespace(
                model=args.model,
                vae_device=args.vae_device,
                dtype=getattr(torch, args.dtype),
                vae_use_slicing=False,
                vae_use_tiling=False,
            )
            worker = VAEWorkerProc(config, None, None)
            with torch.inference_mode():
                actual = worker.decode(VAEDecodeTask("p0-parity", local, args.height, args.width)).output
                # Invoke the existing inline pipeline's decode implementation
                # without allocating its text encoder or DiT transformer.
                inline = SimpleNamespace(
                    vae=worker.vae,
                    vae_scale_factor=worker.vae_scale_factor,
                    _unpack_latents=QwenImagePipeline._unpack_latents,
                )
                expected = QwenImagePipeline._decode_latents(
                    inline, _latent(args).to(device), args.height, args.width
                ).output
            torch.npu.synchronize(device)
            torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=args.rtol, atol=args.atol)
            metrics["decode_max_abs_error"] = (actual.float() - expected.float()).abs().max().item()
        report.send({"role": "receiver", "ok": True, **metrics})
    except Exception:
        report.send({"role": "receiver", "ok": False, "error": traceback.format_exc()})
    finally:
        copied.set()
        report.close()


def _run_pair(args):
    ctx = mp.get_context("spawn")
    tasks, copied = ctx.Queue(), ctx.Event()
    processes, readers = [], []
    try:
        for target in (_sender, _receiver):
            reader, writer = ctx.Pipe(duplex=False)
            process = ctx.Process(target=target, args=(args, tasks, copied, writer))
            process.start()
            writer.close()
            processes.append(process)
            readers.append(reader)
        for reader in readers:
            if not reader.poll(args.timeout):
                raise TimeoutError("Timed out waiting for IPC smoke worker.")
            result = reader.recv()
            print(json.dumps(result), flush=True)
            if not result["ok"]:
                raise RuntimeError(result["error"])
        for process in processes:
            process.join(30)
            if process.exitcode != 0:
                raise RuntimeError(f"Smoke worker exit code: {process.exitcode}")
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(30)
        for reader in readers:
            reader.close()
        tasks.close()
        tasks.join_thread()


async def _run_overlap(args):
    from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    config = OmniDiffusionConfig(
        model=args.model,
        model_class_name="QwenImagePipeline",
        dtype=getattr(torch, args.dtype),
        parallel_config=DiffusionParallelConfig(tensor_parallel_size=args.dit_devices),
        step_execution=True,
        enable_vae_disaggregation=True,
        vae_device=args.vae_device,
        max_num_seqs=1,
        enforce_eager=True,
        profiler_config=(
            {"profiler": "torch", "torch_profiler_dir": str(Path(args.trace_dir).resolve())} if args.trace_dir else None
        ),
    )
    # Match stage startup, which enriches the config before creating the engine.
    config.enrich_config()
    engine = DiffusionEngine.make_engine(config)
    try:
        if args.trace_dir:
            engine.profile(is_start=True)
        await engine._check_and_start_background_loop()

        async def submit(index):
            request = OmniDiffusionRequest(
                prompts=[{"prompt": f"A photograph of a red cup on a table, view {index}."}],
                request_ids=[f"p0-{index}"],
                sampling_params=OmniDiffusionSamplingParams(
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.steps,
                    num_outputs_per_prompt=1,
                    seed=42 + index,
                ),
            )
            output = await engine.async_add_req_and_wait_for_response(request)
            if output.error or output.output is None:
                raise RuntimeError(output.error or "Missing decoded output")
            print(
                json.dumps(
                    {
                        "req_id": f"p0-{index}",
                        "finished_ns": time.monotonic_ns(),
                        "shape": list(output.output.shape),
                        "stages": output.stage_durations,
                    }
                ),
                flush=True,
            )

        await asyncio.wait_for(asyncio.gather(*(submit(i) for i in range(args.requests))), args.timeout)
        if engine.scheduler.get_decoding_req_ids() or engine._out_queue:
            raise RuntimeError("Engine retained pending requests after completion.")
        print("All requests completed. Inspect device traces to establish actual overlap.", flush=True)
    finally:
        if args.trace_dir:
            with suppress(RuntimeError):
                engine.profile(is_start=False)
        engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("ipc", "decode", "overlap"), default="ipc")
    parser.add_argument("--model", help="Local Qwen-Image-2512 directory or Hugging Face model ID")
    parser.add_argument("--dit-device", type=int, default=0, help="Source device for ipc/decode modes")
    parser.add_argument("--dit-devices", type=int, default=1, help="DiT TP size for overlap mode, using devices 0..N-1")
    parser.add_argument("--vae-device", type=int, default=1)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--trace-dir", help="Collect DiT and VAE device traces in overlap mode")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()
    if args.mode != "ipc" and not args.model:
        parser.error("--model is required for decode/overlap")
    if min(args.height, args.width) <= 0 or args.height % 16 or args.width % 16:
        parser.error("height and width must be positive multiples of 16")
    if args.dit_devices < 1 or args.requests < 2 or args.steps < 1 or args.timeout <= 0:
        parser.error("require dit-devices >= 1, requests >= 2, steps >= 1, timeout > 0")
    import torch_npu

    count = torch.npu.device_count()
    if min(args.dit_device, args.vae_device) < 0 or max(args.dit_device, args.vae_device) >= count:
        parser.error(f"devices must be within the {count} visible NPUs")
    if args.mode == "overlap":
        if args.vae_device < args.dit_devices or args.dit_devices > count:
            parser.error("VAE must use a visible device outside DiT devices 0..N-1")
    elif args.dit_device == args.vae_device:
        parser.error("DiT/source and VAE must use different NPUs")
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "torch_npu": torch_npu.__version__,
                "ASCEND_RT_VISIBLE_DEVICES": os.getenv("ASCEND_RT_VISIBLE_DEVICES"),
                "ASCEND_HOME_PATH": os.getenv("ASCEND_HOME_PATH"),
                "visible_npus": count,
            }
        ),
        flush=True,
    )
    if args.mode == "overlap":
        asyncio.run(_run_overlap(args))
    else:
        _run_pair(args)


if __name__ == "__main__":
    main()
