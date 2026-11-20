# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone VAE decode process and its IPC payloads."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.ipc import pack_diffusion_output_shm
from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from diffusers.models.autoencoders import AutoencoderKLQwenImage

logger = init_logger(__name__)


@dataclass
class VAEDecodeTask:
    """Minimal state needed to decode one Qwen-Image latent."""

    req_id: str
    latents: torch.Tensor
    height: int
    width: int
    output_type: str = "pil"
    stage_durations: dict[str, float] = field(default_factory=dict)


@dataclass
class VAEDecodeResult:
    """Result envelope returned from the VAE process to the engine."""

    req_id: str
    result: DiffusionOutput | None = None
    error: str | None = None


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
        vae = AutoencoderKLQwenImage.from_pretrained(
            model,
            subfolder="vae",
            local_files_only=local_files_only,
        ).to(self.device)
        vae.use_slicing = bool(self.od_config.vae_use_slicing)
        vae.use_tiling = bool(self.od_config.vae_use_tiling)
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
        if task.output_type == "latent":
            return DiffusionOutput(output=task.latents, stage_durations=task.stage_durations)

        latents = task.latents.to(device=self.device, dtype=self.vae.dtype)
        latents = self._unpack_latents(latents, task.height, task.width, self.vae_scale_factor)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean
        with torch.inference_mode():
            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
        return DiffusionOutput(output=image, stage_durations=task.stage_durations)

    def busy_loop(self) -> None:
        while True:
            task = self.task_queue.get()
            if task is None:
                return
            if not isinstance(task, VAEDecodeTask):
                logger.error("VAE worker received unexpected task type %r", type(task))
                continue

            logger.info("VAE_DECODE_BEGIN req_id=%s", task.req_id)
            try:
                decode_start = time.perf_counter()
                output = self.decode(task)
                current_omni_platform.synchronize()
                output.stage_durations["vae_decode"] = time.perf_counter() - decode_start
                pack_diffusion_output_shm(output)
                result = VAEDecodeResult(req_id=task.req_id, result=output)
                logger.info("VAE_DECODE_END req_id=%s", task.req_id)
            except Exception as exc:
                logger.exception("VAE decode failed for req_id=%s", task.req_id)
                result = VAEDecodeResult(req_id=task.req_id, error=str(exc))
            self.result_queue.put(result)

    @staticmethod
    def worker_main(
        od_config: OmniDiffusionConfig,
        task_queue: mp.Queue,
        result_queue: mp.Queue,
        ready_writer: Connection,
    ) -> None:
        try:
            worker = VAEWorkerProc(od_config, task_queue, result_queue)
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


__all__ = ["VAEDecodeResult", "VAEDecodeTask", "VAEWorkerProc"]
