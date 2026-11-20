# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone VAE decode process and its IPC payloads."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.ipc import pack_diffusion_output_shm
from vllm_omni.diffusion.worker.vae_ipc import NPUTensorIPC, copy_npu_tensor, ipc_from_wire, ipc_to_wire
from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from diffusers.models.autoencoders import AutoencoderKLQwenImage

logger = init_logger(__name__)


@dataclass
class VAEDecodeTask:
    """Minimal state needed to decode one Qwen-Image latent."""

    req_id: str
    latents: torch.Tensor | None
    height: int
    width: int
    output_type: str = "pil"
    stage_durations: dict[str, float] = field(default_factory=dict)
    latent_ipc: NPUTensorIPC | None = None


@dataclass
class VAEDecodeResult:
    """Result envelope returned from the VAE process to the engine."""

    req_id: str
    result: DiffusionOutput | None = None
    error: str | None = None


def encode_vae_task(task: VAEDecodeTask) -> str:
    if task.latents is not None or task.latent_ipc is None:
        raise ValueError("Only descriptor-only NPU tasks may use the JSON transport.")
    return json.dumps(
        {
            "req_id": task.req_id,
            "height": task.height,
            "width": task.width,
            "output_type": task.output_type,
            "stage_durations": task.stage_durations,
            "latent_ipc": ipc_to_wire(task.latent_ipc),
        }
    )


def decode_vae_task(message: str) -> VAEDecodeTask:
    data = json.loads(message)
    payload = ipc_from_wire(data.pop("latent_ipc"))
    return VAEDecodeTask(latents=None, latent_ipc=payload, **data)


class VAEWorkerProc:
    """Own a VAE module and service decode tasks in a separate process."""

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        task_queue: mp.Queue,
        result_queue: mp.Queue,
    ) -> None:
        self.od_config = od_config
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.device = self._init_device()
        self.vae = self._load_vae()
        self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample)

    def _init_device(self) -> torch.device:
        device_index = int(getattr(self.od_config, "vae_device", 0))
        os.environ["LOCAL_RANK"] = str(device_index)
        device = current_omni_platform.get_torch_device(device_index)
        current_omni_platform.set_device(device)
        return device

    def _load_vae(self) -> AutoencoderKLQwenImage:
        # The distributed wrapper requires a DiT process group.  A standalone
        # VAE worker deliberately loads the underlying local diffusers module.
        from diffusers.models.autoencoders import AutoencoderKLQwenImage

        model = self.od_config.model
        local_files_only = os.path.exists(model)
        # Match the dtype context used by DiffusersPipelineLoader when it
        # constructs the inline Qwen pipeline; preserve Diffusers' own policy.
        with set_default_torch_dtype(self.od_config.dtype):
            vae = AutoencoderKLQwenImage.from_pretrained(
                model,
                subfolder="vae",
                local_files_only=local_files_only,
            ).to(self.device)
        vae.use_slicing = bool(self.od_config.vae_use_slicing)
        vae.use_tiling = bool(self.od_config.vae_use_tiling)
        logger.info("VAE_MODEL_READY device=%s dtype=%s", self.device, vae.dtype)
        return vae.eval()

    @staticmethod
    def _unpack_latents(
        latents: torch.Tensor,
        height: int,
        width: int,
        vae_scale_factor: int,
    ) -> torch.Tensor:
        batch_size, _, channels = latents.shape
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))
        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        return latents.reshape(batch_size, channels // 4, 1, height, width)

    def decode(self, task: VAEDecodeTask) -> DiffusionOutput:
        copy_start = time.perf_counter()
        if task.latent_ipc is not None:
            latents = copy_npu_tensor(task.latent_ipc, self.device, self.vae.dtype)
        elif task.latents is not None:
            latents = task.latents.to(device=self.device, dtype=self.vae.dtype)
        else:
            raise ValueError("VAE task has neither latents nor an IPC descriptor.")
        current_omni_platform.synchronize()
        task.stage_durations["vae_input_copy"] = time.perf_counter() - copy_start
        logger.info("VAE_LOCAL_COPY_END req_id=%s device=%s ts_ns=%d", task.req_id, self.device, time.monotonic_ns())
        if task.output_type == "latent":
            return DiffusionOutput(output=latents, stage_durations=task.stage_durations)

        latents = self._unpack_latents(latents, task.height, task.width, self.vae_scale_factor)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        current_omni_platform.synchronize()
        decode_start = time.perf_counter()
        logger.info("VAE_DECODE_BEGIN req_id=%s device=%s ts_ns=%d", task.req_id, self.device, time.monotonic_ns())
        with torch.inference_mode(), torch.profiler.record_function(f"vae_decode:{task.req_id}"):
            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
        current_omni_platform.synchronize()
        task.stage_durations["vae_decode"] = time.perf_counter() - decode_start
        logger.info("VAE_DECODE_END req_id=%s device=%s ts_ns=%d", task.req_id, self.device, time.monotonic_ns())
        return DiffusionOutput(output=image, stage_durations=task.stage_durations)

    def busy_loop(self) -> None:
        while True:
            task = self.task_queue.get()
            if task is None:
                return
            # NPU descriptors are serialized on the producer thread, so a
            # serialization failure becomes a request error before DECODING.
            if isinstance(task, str):
                req_id = json.loads(task)["req_id"]
                try:
                    task = decode_vae_task(task)
                except Exception as exc:
                    logger.exception("VAE IPC metadata invalid for req_id=%s", req_id)
                    self.result_queue.put(VAEDecodeResult(req_id=req_id, error=f"VAE IPC metadata invalid: {exc}"))
                    continue
            if not isinstance(task, VAEDecodeTask):
                logger.error("VAE worker received unexpected task type %r", type(task))
                continue

            logger.info("VAE_RECEIVE req_id=%s ts_ns=%d", task.req_id, time.monotonic_ns())
            try:
                output = self.decode(task)
                current_omni_platform.synchronize()
                # Always return CPU data, including outputs below the SHM
                # threshold; the engine need not initialize a VAE NPU context.
                if isinstance(output.output, torch.Tensor):
                    output.output = output.output.detach().cpu()
                pack_diffusion_output_shm(output)
                result = VAEDecodeResult(req_id=task.req_id, result=output)
            except Exception as exc:
                logger.exception("VAE decode failed for req_id=%s", task.req_id)
                result = VAEDecodeResult(req_id=task.req_id, error=str(exc))
            self.result_queue.put(result)
            del task, result

    @staticmethod
    def worker_main(
        od_config: OmniDiffusionConfig,
        task_queue: mp.Queue,
        result_queue: mp.Queue,
        ready_writer: Connection,
    ) -> None:
        profiler = None
        try:
            worker = VAEWorkerProc(od_config, task_queue, result_queue)
            if getattr(od_config.profiler_config, "profiler", None) == "torch":
                from vllm_omni.profiler import create_omni_profiler

                profiler = create_omni_profiler(
                    od_config.profiler_config, worker_name="vae_worker", local_rank=od_config.vae_device
                )
                profiler.start()
            ready_writer.send({"status": "ready"})
            worker.busy_loop()
        except Exception as exc:
            logger.exception("VAE worker initialization or loop failed")
            try:
                ready_writer.send({"status": "error", "error": str(exc)})
            except (BrokenPipeError, EOFError, OSError):
                pass
        finally:
            ready_writer.close()
            # Wake a blocking engine-side collector on both normal and
            # unexpected VAE process exits.
            try:
                result_queue.put(None)
            except (OSError, ValueError):
                pass
            if profiler is not None:
                profiler.stop()


__all__ = ["VAEDecodeResult", "VAEDecodeTask", "VAEWorkerProc"]
