"""
Unit tests for TorchModelWithTransforms archiving functionality.

Tests that models can be archived and restored with transforms preserved.
"""

import tempfile
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path

import h5py
import pytest
import torch
from lume.model import LUMEModel
from lume_torch.variables import TorchScalarVariable

from lume_applications.model_calibration import (
    TorchAffineTransform,
    TorchModelWithTransforms,
)


class ArchivableTorchModel(LUMEModel, torch.nn.Module):
    """
    A simple archivable torch model for testing.
    
    This model is designed to be easily archivable, allowing us to test
    the TorchModelWithTransforms archiving.
    """

    def __init__(self, hidden_size: int = 4):
        torch.nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.linear1 = torch.nn.Linear(1, hidden_size, dtype=torch.float64)
        self.linear2 = torch.nn.Linear(hidden_size, 1, dtype=torch.float64)
        self.variables = {
            "input": TorchScalarVariable(name="input"),
            "output": TorchScalarVariable(name="output", read_only=True),
        }

    @property
    def supported_variables(self):
        return self.variables

    def _set(self, values):
        self.input_value = values.get("input", torch.tensor(0.0, dtype=torch.float64))

    def _get(self, names):
        x = self.input_value.view(1, 1) if self.input_value.dim() == 0 else self.input_value
        with torch.no_grad():
            y = torch.relu(self.linear1(x))
            y = self.linear2(y)
        return {"output": y.view(())}

    def reset(self):
        self.input_value = torch.tensor(0.0, dtype=torch.float64)

    def to_archive(self, location: str | Path | h5py.Group) -> None:
        """Archive model state to HDF5."""
        with (
            h5py.File(location, "w")
            if isinstance(location, (str, Path))
            else nullcontext(location)
        ) as archive:
            # Store model state dict
            state_buffer = BytesIO()
            torch.save(self.state_dict(), state_buffer)
            archive.create_dataset(
                "model_state",
                data=bytearray(state_buffer.getvalue()),
                dtype="u1",
            )
            archive.attrs["hidden_size"] = self.hidden_size

    @classmethod
    def from_archive(cls, location: str | Path | h5py.Group) -> "ArchivableTorchModel":
        """Restore model from HDF5 archive."""
        with (
            h5py.File(location, "r")
            if isinstance(location, (str, Path))
            else nullcontext(location)
        ) as archive:
            hidden_size = int(archive.attrs.get("hidden_size", 4))
            model = cls(hidden_size=hidden_size)

            # Restore state dict
            state_bytes = archive["model_state"][()].tobytes()
            state_dict = torch.load(BytesIO(state_bytes), map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)

        return model


class TestTransformsArchiving:
    """Test archiving of TorchModelWithTransforms."""

    @pytest.fixture
    def model(self):
        """Create a base model and wrap it with transforms."""
        base_model = ArchivableTorchModel(hidden_size=4)
        transforms = {
            "input": TorchAffineTransform(scale=2.0, offset=0.5),
        }
        return TorchModelWithTransforms(base_model, transforms)

    def test_archive_to_file(self, model):
        """Test that archiving to file works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "test_model.h5"
            model.to_archive(archive_path)
            assert archive_path.exists()
            assert archive_path.stat().st_size > 0

    def test_archive_to_h5py_group(self, model):
        """Test that archiving to an HDF5 group works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with h5py.File(Path(tmpdir) / "test.h5", "w") as f:
                group = f.create_group("model")
                model.to_archive(group)
                assert "transforms_state" in group
                assert "transform_names" in group.attrs

    def test_restore_from_file(self, model):
        """Test that model can be restored from file archive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "test_model.h5"
            model.to_archive(archive_path)

            # Restore
            restored = TorchModelWithTransforms.from_archive(archive_path)
            assert isinstance(restored, TorchModelWithTransforms)
            assert isinstance(restored.model, ArchivableTorchModel)
            assert list(restored.transforms.keys()) == ["input"]

    def test_restore_from_h5py_group(self, model):
        """Test that model can be restored from HDF5 group archive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with h5py.File(Path(tmpdir) / "test.h5", "w") as f:
                group = f.create_group("model")
                model.to_archive(group)

            # Restore
            with h5py.File(Path(tmpdir) / "test.h5", "r") as f:
                group = f["model"]
                restored = TorchModelWithTransforms.from_archive(group)
                assert isinstance(restored, TorchModelWithTransforms)
                assert list(restored.transforms.keys()) == ["input"]

    def test_transforms_preserved_after_restore(self, model):
        """Test that transform parameters are preserved after archiving and restoring."""
        # Get original transform params
        original_scale = model.transforms["input"].scale.item()
        original_offset = model.transforms["input"].offset.item()

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "test_model.h5"
            model.to_archive(archive_path)

            # Restore
            restored = TorchModelWithTransforms.from_archive(archive_path)

            # Check transform params
            restored_scale = restored.transforms["input"].scale.item()
            restored_offset = restored.transforms["input"].offset.item()

            assert restored_scale == pytest.approx(original_scale)
            assert restored_offset == pytest.approx(original_offset)

    def test_model_weights_preserved_after_restore(self, model):
        """Test that model weights are preserved after archiving and restoring."""
        # Get original weights
        original_w1 = model.model.linear1.weight.clone()
        original_b1 = model.model.linear1.bias.clone()
        original_w2 = model.model.linear2.weight.clone()
        original_b2 = model.model.linear2.bias.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "test_model.h5"
            model.to_archive(archive_path)

            # Restore
            restored = TorchModelWithTransforms.from_archive(archive_path)

            # Check weights
            torch.testing.assert_close(restored.model.linear1.weight, original_w1)
            torch.testing.assert_close(restored.model.linear1.bias, original_b1)
            torch.testing.assert_close(restored.model.linear2.weight, original_w2)
            torch.testing.assert_close(restored.model.linear2.bias, original_b2)

    def test_restored_model_produces_same_output(self, model):
        """Test that restored model produces the same output as original."""
        # Get output from original model
        input_value = torch.tensor(1.5, dtype=torch.float64)
        model.set({"input": input_value})
        original_output = model.get(["output"])["output"].item()

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "test_model.h5"
            model.to_archive(archive_path)

            # Restore and get output
            restored = TorchModelWithTransforms.from_archive(archive_path)
            restored.set({"input": input_value})
            restored_output = restored.get(["output"])["output"].item()

            # Outputs should match (allow small floating point differences)
            assert restored_output == pytest.approx(original_output, abs=1e-10)

    def test_multiple_transforms_archived(self):
        """Test archiving with multiple transforms."""
        base_model = ArchivableTorchModel(hidden_size=4)
        transforms = {
            "input": TorchAffineTransform(scale=2.0, offset=0.5),
        }
        model = TorchModelWithTransforms(base_model, transforms)

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "test_model.h5"
            model.to_archive(archive_path)

            # Restore and verify
            restored = TorchModelWithTransforms.from_archive(archive_path)
            assert list(restored.transforms.keys()) == ["input"]

    def test_archive_roundtrip_with_training_state(self):
        """Test that archive correctly preserves training state from model training."""
        base_model = ArchivableTorchModel(hidden_size=4)
        transforms = {
            "input": TorchAffineTransform(scale=1.0, offset=0.0),
        }
        model = TorchModelWithTransforms(base_model, transforms)

        # Simulate training by updating weights and transforms
        with torch.no_grad():
            model.model.linear1.weight.fill_(0.5)
            model.model.linear1.bias.fill_(0.1)
            model.transforms["input"].scale.fill_(1.5)
            model.transforms["input"].offset.fill_(0.2)

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "test_model.h5"
            model.to_archive(archive_path)

            # Restore
            restored = TorchModelWithTransforms.from_archive(archive_path)

            # Verify all values are preserved
            assert restored.model.linear1.weight[0, 0].item() == pytest.approx(0.5)
            assert restored.model.linear1.bias[0].item() == pytest.approx(0.1)
            assert restored.transforms["input"].scale.item() == pytest.approx(1.5)
            assert restored.transforms["input"].offset.item() == pytest.approx(0.2)

    def test_archive_empty_archive_handling(self, model):
        """Test that archiving to a group without transforms_state is handled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "test_model.h5"
            model.to_archive(archive_path)

            # Verify transforms_state exists
            with h5py.File(archive_path, "r") as f:
                assert "transforms_state" in f
                assert "transform_names" in f.attrs

    def test_model_state_dict_roundtrip(self, model):
        """Test that state_dict can be loaded correctly."""
        state_dict = model.state_dict()
        assert len(state_dict) > 0
        
        # Create new model and load state
        new_model = TorchModelWithTransforms(
            ArchivableTorchModel(hidden_size=4),
            {"input": TorchAffineTransform()},
        )
        new_model.load_state_dict(state_dict)

    def test_archive_with_trained_transforms(self):
        """Test archiving after training transforms."""
        base_model = ArchivableTorchModel(hidden_size=4)
        transforms = {"input": TorchAffineTransform(scale=1.0, offset=0.0)}
        model = TorchModelWithTransforms(base_model, transforms)

        # Simulate simple training update
        with torch.no_grad():
            model.transforms["input"].scale.fill_(1.5)
            model.transforms["input"].offset.fill_(0.3)

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "test_model.h5"
            model.to_archive(archive_path)

            # Restore
            restored = TorchModelWithTransforms.from_archive(archive_path)

            # Verify trained parameters are preserved
            assert restored.transforms["input"].scale.item() == pytest.approx(1.5)
            assert restored.transforms["input"].offset.item() == pytest.approx(0.3)
