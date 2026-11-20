# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TorchNPU IPC descriptors for the standalone VAE data plane.

The queue carries descriptors, never the NPU tensor itself. Import on the VAE
device, then copy into local HBM before releasing the imported storage. See
Ascend/pytorch docs/zh/developer_notes/memory_management/memory_sharing_ipc.md.
"""

from __future__ import annotations

import base64
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class NPUTensorIPC:
    rebuild: Callable[..., torch.Tensor]
    args: tuple[Any, ...]
    source_device: int


_TORCH_TYPES = {
    "Tensor": torch.Tensor,
    "TypedStorage": torch.storage.TypedStorage,
    "UntypedStorage": torch.UntypedStorage,
}


def _encode_arg(value: Any) -> Any:
    for name, cls in _TORCH_TYPES.items():
        if value is cls:
            return {"torch_type": name}
    if isinstance(value, torch.dtype):
        return {"dtype": str(value).removeprefix("torch.")}
    if isinstance(value, bytes):
        return {"bytes": base64.b64encode(value).decode("ascii")}
    if isinstance(value, (tuple, list)):
        return [_encode_arg(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Unsupported NPU IPC metadata type: {type(value).__name__}")


def _decode_arg(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_decode_arg(item) for item in value)
    if isinstance(value, dict):
        if set(value) == {"torch_type"}:
            return _TORCH_TYPES[value["torch_type"]]
        if set(value) == {"dtype"}:
            dtype = getattr(torch, value["dtype"], None)
            if isinstance(dtype, torch.dtype):
                return dtype
        if set(value) == {"bytes"}:
            return base64.b64decode(value["bytes"], validate=True)
        raise ValueError("Invalid NPU IPC metadata tag.")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ValueError("Invalid NPU IPC metadata value.")


def ipc_to_wire(payload: NPUTensorIPC) -> dict[str, Any]:
    """JSON-compatible, data-only metadata; no serialized executable objects."""
    return {
        "parameters": list(inspect.signature(payload.rebuild).parameters),
        "args": [_encode_arg(arg) for arg in payload.args],
        "source_device": payload.source_device,
    }


def ipc_from_wire(data: dict[str, Any]) -> NPUTensorIPC:
    from torch_npu.multiprocessing.reductions import rebuild_npu_tensor

    if data["parameters"] != list(inspect.signature(rebuild_npu_tensor).parameters):
        raise RuntimeError("Producer and VAE TorchNPU rebuild signatures differ.")
    return NPUTensorIPC(rebuild_npu_tensor, tuple(_decode_arg(arg) for arg in data["args"]), data["source_device"])


def _target_args(payload: NPUTensorIPC, device_index: int) -> tuple[Any, ...]:
    """Locate the device argument by name; fail closed on unsupported versions."""
    parameters = list(inspect.signature(payload.rebuild).parameters.values())
    names = [p.name for p in parameters]
    if (
        len(parameters) != len(payload.args)
        or "storage_device" not in names
        or any(p.kind not in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in parameters)
    ):
        raise RuntimeError("Unsupported TorchNPU rebuild_npu_tensor signature; check the installed torch_npu version.")
    args = list(payload.args)
    args[names.index("storage_device")] = device_index
    return tuple(args)


def export_npu_tensor(tensor: torch.Tensor) -> NPUTensorIPC:
    if tensor.device.type != "npu":
        raise ValueError("NPU IPC export requires an NPU tensor.")
    # Import registers TorchNPU's multiprocessing reducers. Look up the reducer
    # afterwards, rather than capturing PyTorch's unpatched function at import.
    import torch_npu  # noqa: F401

    tensor = tensor.detach().contiguous()
    # A conservative boundary for P0, also supporting reducers without IPC
    # events. This waits only for the producer device, not for VAE decode.
    torch.npu.synchronize(tensor.device)
    rebuild, args = torch.multiprocessing.reductions.reduce_tensor(tensor)
    if rebuild.__name__ != "rebuild_npu_tensor" or not rebuild.__module__.startswith("torch_npu."):
        raise RuntimeError("TorchNPU NPU IPC reducer is not registered; check the installed torch_npu version.")
    payload = NPUTensorIPC(rebuild, tuple(args), tensor.device.index)
    _target_args(payload, tensor.device.index)
    return payload


def copy_npu_tensor(payload: NPUTensorIPC, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Import peer storage and return an independent tensor in local HBM."""
    if device.type != "npu" or device.index is None:
        raise ValueError("NPU IPC import requires an explicitly indexed NPU device.")
    import torch_npu  # noqa: F401

    with torch.npu.device(device):
        shared = payload.rebuild(*_target_args(payload, device.index))
        try:
            # .to(device) can return the imported alias unchanged. An explicit
            # allocation is required to obtain VAE-local ownership.
            local = torch.empty(shared.shape, device=device, dtype=dtype)
            local.copy_(shared)
        finally:
            # Do not release the IPC mapping while an asynchronous copy uses it.
            torch.npu.synchronize(device)
            del shared
    return local
