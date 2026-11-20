# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests; real peer access is checked by vae_disaggregation_smoke.py."""

import json
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from vllm_omni.diffusion.worker.vae_ipc import (
    NPUTensorIPC,
    _target_args,
    copy_npu_tensor,
    export_npu_tensor,
    ipc_from_wire,
    ipc_to_wire,
)


def _rebuild(tensor, storage_device):
    return tensor


class TestVAEIPC(unittest.TestCase):
    def test_wire_metadata_roundtrip_has_only_data(self):
        def rebuild_npu_tensor(tensor_cls, storage_device, storage_cls, dtype, handle, size):
            return None

        payload = NPUTensorIPC(
            rebuild_npu_tensor,
            (torch.Tensor, 0, torch.storage.TypedStorage, torch.bfloat16, b"\x00\xff", (1, 4, 64)),
            0,
        )
        wire = json.loads(json.dumps(ipc_to_wire(payload)))
        with patch.dict(
            "sys.modules",
            {"torch_npu.multiprocessing.reductions": SimpleNamespace(rebuild_npu_tensor=rebuild_npu_tensor)},
        ):
            restored = ipc_from_wire(wire)
        self.assertEqual(restored.args, payload.args)
        self.assertEqual(restored.source_device, 0)
        self.assertIs(restored.rebuild, rebuild_npu_tensor)

    def test_wire_rejects_tensor_data(self):
        with self.assertRaisesRegex(TypeError, "metadata type"):
            ipc_to_wire(NPUTensorIPC(_rebuild, (torch.zeros(2), 0), 0))

    def test_wire_rejects_different_receiver_signature(self):
        with (
            patch.dict(
                "sys.modules", {"torch_npu.multiprocessing.reductions": SimpleNamespace(rebuild_npu_tensor=_rebuild)}
            ),
            self.assertRaisesRegex(RuntimeError, "signatures differ"),
        ):
            ipc_from_wire({"parameters": ["unknown"], "args": [], "source_device": 0})

    def test_target_device_is_located_by_parameter_name(self):
        tensor = torch.arange(6).reshape(2, 3)
        payload = NPUTensorIPC(_rebuild, (tensor, 0), 0)
        args = _target_args(payload, 3)
        self.assertIs(args[0], tensor)
        self.assertEqual(args[1], 3)
        self.assertEqual(payload.args[1], 0)

    def test_unknown_signature_is_rejected(self):
        for rebuild, args in [(lambda tensor, device: tensor, (None, 0)), (_rebuild, (None,))]:
            with self.subTest(rebuild=rebuild), self.assertRaisesRegex(RuntimeError, "signature"):
                _target_args(NPUTensorIPC(rebuild, args, 0), 1)

    def test_cpu_export_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "NPU tensor"):
            export_npu_tensor(torch.zeros(2))

    def test_export_checks_readiness_and_uses_registered_reducer(self):
        tensor = Mock()
        tensor.device = SimpleNamespace(type="npu", index=0)
        tensor.detach.return_value.contiguous.return_value = tensor

        def rebuild_npu_tensor(tensor, storage_device):
            return tensor

        rebuild_npu_tensor.__module__ = "torch_npu.multiprocessing.reductions"
        reducer = Mock(return_value=(rebuild_npu_tensor, (None, 0)))
        synchronize = Mock()
        with (
            patch.dict("sys.modules", {"torch_npu": SimpleNamespace()}),
            patch.object(torch, "npu", SimpleNamespace(synchronize=synchronize), create=True),
            patch.object(torch.multiprocessing.reductions, "reduce_tensor", reducer),
        ):
            payload = export_npu_tensor(tensor)
        synchronize.assert_called_once_with(tensor.device)
        reducer.assert_called_once_with(tensor)
        self.assertIs(payload.rebuild, rebuild_npu_tensor)
        self.assertEqual(payload.source_device, 0)

    def test_unregistered_reducer_fails_before_submission(self):
        tensor = Mock()
        tensor.device = SimpleNamespace(type="npu", index=0)
        tensor.detach.return_value.contiguous.return_value = tensor
        with (
            patch.dict("sys.modules", {"torch_npu": SimpleNamespace()}),
            patch.object(torch, "npu", SimpleNamespace(synchronize=Mock()), create=True),
            patch.object(torch.multiprocessing.reductions, "reduce_tensor", return_value=(_rebuild, (None, 0))),
            self.assertRaisesRegex(RuntimeError, "not registered"),
        ):
            export_npu_tensor(tensor)

    def test_import_requires_indexed_npu(self):
        payload = NPUTensorIPC(_rebuild, (None, 0), 0)
        with self.assertRaisesRegex(ValueError, "indexed NPU"):
            copy_npu_tensor(payload, torch.device("cpu"), torch.float32)

    def test_local_copy_owns_storage_and_synchronizes(self):
        tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        imported_devices = []

        def rebuild(tensor, storage_device):
            imported_devices.append(storage_device)
            return tensor

        npu = SimpleNamespace(device=lambda _: nullcontext(), synchronize=Mock())
        local = torch.empty_like(tensor)
        with (
            patch.dict("sys.modules", {"torch_npu": SimpleNamespace()}),
            patch.object(torch, "npu", npu, create=True),
            patch.object(torch, "empty", return_value=local),
        ):
            result = copy_npu_tensor(
                NPUTensorIPC(rebuild, (tensor, 0), 0),
                SimpleNamespace(type="npu", index=1),
                torch.float32,
            )
        self.assertEqual(imported_devices, [1])
        self.assertTrue(torch.equal(result, tensor))
        self.assertNotEqual(result.data_ptr(), tensor.data_ptr())
        npu.synchronize.assert_called_once()
        tensor.zero_()
        self.assertEqual(result.sum().item(), 15)

    def test_failed_copy_still_waits_before_releasing_import(self):
        npu = SimpleNamespace(device=lambda _: nullcontext(), synchronize=Mock())
        local = Mock()
        local.copy_.side_effect = RuntimeError("copy failed")
        with (
            patch.dict("sys.modules", {"torch_npu": SimpleNamespace()}),
            patch.object(torch, "npu", npu, create=True),
            patch.object(torch, "empty", return_value=local),
            self.assertRaisesRegex(RuntimeError, "copy failed"),
        ):
            copy_npu_tensor(
                NPUTensorIPC(_rebuild, (torch.ones(2), 0), 0),
                SimpleNamespace(type="npu", index=1),
                torch.float32,
            )
        npu.synchronize.assert_called_once()


if __name__ == "__main__":
    unittest.main()
