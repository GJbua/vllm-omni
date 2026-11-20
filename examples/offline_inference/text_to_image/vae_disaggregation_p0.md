# DiT/VAE disaggregation: P0 verification on Ascend

Run from the repository root using the installed vLLM-Omni Ascend environment.
No batching knobs or sampling-key changes are required. The VAE worker is a
sibling of the DiT workers, owned by their executor. `num_gpus` counts DiT ranks
only. `vae_device` is an index into the stage's visible devices, not a physical
device ID. For explicit stage YAML, include both DiT and VAE cards in
`runtime.devices`; the two default configuration factories now include the VAE.

First record `npu-smi info`, the installed HDK/CANN versions and the Git commit.
The script reports Torch/TorchNPU versions and visibility. Confirm that the
installed versions support the [official TorchNPU IPC recipe](https://github.com/Ascend/pytorch/blob/master/docs/zh/developer_notes/memory_management/memory_sharing_ipc.md).
That document currently requires HDK >= 25.3.RC1 and CANN >= 8.3.RC1; check the
documentation for your installed TorchNPU release as well.

## 1. Pure Tensor IPC, two spawned processes and two NPUs

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
python examples/offline_inference/text_to_image/vae_disaggregation_smoke.py --mode ipc
```

Expected: both workers report `ok: true`; the local copy matches the source
bit-for-bit. The source process stays alive until the receiver finishes using
the peer storage. Export uses TorchNPU's registered reducer. Import locates
`storage_device` in the rebuild signature rather than assuming a tuple index.
An unsupported reducer fails explicitly, without falling back through CPU.

## 2. Fixed-latent decode parity, VAE weights only

```bash
python examples/offline_inference/text_to_image/vae_disaggregation_smoke.py \
  --mode decode --model /path/to/Qwen-Image-2512 --height 512 --width 512
```

Expected: IPC equality and decode parity pass. This invokes the original
`QwenImagePipeline._decode_latents` as reference, sharing the same VAE weights,
dtype, slicing and tiling settings. It isolates the IPC/decode contract and
does not load a full DiT or validate denoising quality. Report the measured
maximum error and the supplied tolerances; defaults are `rtol=atol=1e-3`.

## 3. Real Engine, DECODING completion and overlap traces

```bash
python examples/offline_inference/text_to_image/vae_disaggregation_smoke.py \
  --mode overlap --model /path/to/Qwen-Image-2512 \
  --dit-devices 1 --vae-device 1 --requests 3 --steps 20 \
  --trace-dir ./p0_traces 2>&1 | tee p0_overlap.log
```

This runs the actual Engine/Scheduler/Executor/Worker/Runner/Pipeline chain,
including the constructor warmup. All requests must return decoded tensors and
leave no pending Futures or DECODING states. The script timeout applies to
request execution; external startup/job timeouts should also cover model load
and warmup. A single DiT NPU must have enough memory for the model. If DiT needs
two cards, expose three and use `--dit-devices 2 --vae-device 2`.

`--trace-dir` enables the existing platform profiler for DiT and an independent
profiler in the VAE process. VAE tracing includes warmup and stops on shutdown.
Compare device execution intervals across ranks, not merely host log ordering.
`DENOISE_BEGIN/END`, `VAE_SUBMIT/RECEIVE`, `VAE_LOCAL_COPY_END` and
`VAE_DECODE_BEGIN/END` contain same-host monotonic timestamps. A positive device
overlap between A's VAE decode and B's DiT work establishes pipeline overlap;
the script does not claim overlap just because all requests completed.

The P0 path deliberately synchronizes producer readiness and the VAE-local
copy. Shared IPC storage is released only after that copy completes. Outputs,
including small tensors below the shared-memory threshold, return on CPU.
Transport/submit errors fail the corresponding request; process death follows
the executor's existing failure path. Live-but-stalled worker recovery,
batching, multiple independent DiT groups, cancellation of queued VAE tasks,
and production backpressure remain outside this P0 change.

## CPU contract/regression tests (full project dependencies required)

```bash
python -m pytest tests/diffusion/test_vae_ipc.py
python -m pytest tests/diffusion/test_diffusion_step_pipeline.py \
  tests/diffusion/test_diffusion_scheduler.py -m cpu
```
