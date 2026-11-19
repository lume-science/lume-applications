from copy import deepcopy

import numpy as np
import pytest
import torch
from lume.exceptions import ReadOnlyError
from lume.model import LUMEModel
from lume.variables import ScalarVariable
from lume_torch.variables import TorchScalarVariable

from lume_applications.model_calibration import (
    AffineTransform,
    ModelWithTransforms,
    TorchAffineTransform,
    TorchModelWithTransforms,
    calibrate_model,
    get_trainable_parameters,
)


class ScalarModel(LUMEModel):
    def __init__(self):
        self.values = {"x": 0.0, "z": 0.0}
        self.variables = {
            name: ScalarVariable(name=name, value_range=(-10.0, 10.0))
            for name in self.values
        }

    @property
    def supported_variables(self):
        return self.variables

    def _set(self, values):
        self.values.update(values)

    def _get(self, names):
        return {name: self.values[name] for name in names}

    def reset(self):
        self.values = {"x": 0.0, "z": 0.0}


class TensorModel(LUMEModel, torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.curvature = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float64))
        self.variables = {
            name: TorchScalarVariable(name=name)
            for name in ("x", "z", "a", "valid", "invalid.name")
        }
        self.variables["y"] = TorchScalarVariable(name="y", read_only=True)
        self.reset()

    @property
    def supported_variables(self):
        return self.variables

    def _set(self, values):
        self.x = values["x"]

    def _get(self, names):
        return {"y": self.x + self.curvature * self.x.square()}

    def reset(self):
        self.x = self.curvature.new_zeros(())


def test_generic_wrapper_with_non_torch_lume_model():
    model = ScalarModel()
    wrapper = ModelWithTransforms(model, transforms={"x": AffineTransform(2, 1)})
    assert not hasattr(model, "transforms")
    inputs = {"x": 2.0, "z": 3.0}
    wrapper.set(inputs)
    assert model.get(["x", "z"]) == {"x": 5.0, "z": 3.0}
    assert wrapper.get(["x", "z"]) == {"x": 2.0, "z": 3.0}
    assert inputs == {"x": 2.0, "z": 3.0}
    wrapper.set({"z": 4.0})
    assert wrapper.get(["x", "z"]) == {"x": 2.0, "z": 4.0}
    with pytest.raises(ValueError, match="not supported"):
        wrapper.set({"missing": 1.0})
    with pytest.raises(ValueError, match="not supported"):
        wrapper.get(["missing"])
    with pytest.raises(ValueError, match="unknown transform"):
        ModelWithTransforms(ScalarModel(), {"missing": AffineTransform()})
    for invalid in (1, lambda value: value, AffineTransform):
        with pytest.raises(TypeError, match="transform and untransform"):
            ModelWithTransforms(ScalarModel(), {"x": invalid})
    read_only_model = ScalarModel()
    read_only_model.variables["x"].read_only = True
    read_only_model.values["x"] = 5.0
    output_wrapper = ModelWithTransforms(read_only_model, {"x": AffineTransform(2, 1)})
    assert output_wrapper.get_value("x") == 2.0
    assert read_only_model.values["x"] == 5.0
    with pytest.raises(ReadOnlyError):
        output_wrapper.set_value("x", 1.0)
    assert not hasattr(wrapper, "parameters")


def test_lume_model_interface_and_effective_value_validation():
    model = ScalarModel()
    model.variables["x"].default_validation_config = "error"
    wrapper = ModelWithTransforms(model, {"x": AffineTransform(0.5)})
    assert isinstance(wrapper, LUMEModel)
    assert wrapper.supported_variables is model.supported_variables
    wrapper.set_value("x", 12.0)
    assert model.get_value("x") == 6.0
    assert wrapper.get_value("x") == 12.0
    with pytest.raises(ValueError):
        wrapper.set_value("x", 22.0)
    assert wrapper.get_value("x") == 12.0
    transforms = wrapper.transforms
    wrapper.reset()
    assert wrapper.get_value("x") == 0.0
    assert wrapper.transforms is transforms
    wrapper.set_value("x", 4.0)
    assert wrapper.get_value("x") == 4.0
    with pytest.raises(TypeError, match="LUMEModel"):
        ModelWithTransforms(object(), {})
    with pytest.raises(TypeError, match="LUMEModel"):
        TorchModelWithTransforms(torch.nn.Module(), {"x": TorchAffineTransform()})


def test_generic_wrappers_own_independent_mappings():
    model = ScalarModel()
    transforms = {"x": AffineTransform(2, 1)}
    first = ModelWithTransforms(model, transforms)
    second = ModelWithTransforms(model, {"x": AffineTransform(3, 2)})
    transforms.clear()
    first.set_value("x", 1.0)
    assert model.get_value("x") == 3.0
    second.set_value("x", 1.0)
    assert model.get_value("x") == 5.0
    assert first.get_value("x") == 2.0
    assert not hasattr(model, "transforms")


def test_transform_failure_propagates_without_setting_model():
    class FailingTransform(AffineTransform):
        def transform(self, value):
            raise ValueError("invalid calibration input")

        def untransform(self, value):
            raise ValueError("invalid calibration output")

    model = ScalarModel()
    wrapper = ModelWithTransforms(model, {"x": FailingTransform()})
    with pytest.raises(ValueError, match="invalid calibration input"):
        wrapper.set({"z": 3.0, "x": 1.0})
    assert model.values == {"x": 0.0, "z": 0.0}
    with pytest.raises(ValueError, match="invalid calibration output"):
        wrapper.get_value("x")


@pytest.mark.parametrize("scale", [2.0, -2.0])
def test_generic_affine_scalar_and_array_round_trip(scale):
    transform = AffineTransform(scale=scale, offset=1)
    assert not isinstance(transform, torch.nn.Module)
    assert transform.transform(3.0) == scale * 3.0 + 1
    assert transform.untransform(transform.transform(3.0)) == 3.0
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = transform.transform(values)
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, scale * values + 1)
    np.testing.assert_allclose(transform.untransform(result), values)
    transform.reset_to_identity()
    np.testing.assert_allclose(transform.transform(values), values)
    assert transform.get_transform_params() == {"scale": 1.0, "offset": 0.0}


@pytest.mark.parametrize(
    "kwargs", [{"scale": 0}, {"scale": float("nan")}, {"offset": float("inf")}]
)
def test_generic_affine_rejects_noninvertible_or_nonfinite_coefficients(kwargs):
    with pytest.raises(ValueError):
        AffineTransform(**kwargs)


@pytest.mark.parametrize("scale", [2.0, -2.0])
def test_affine_transform_shapes_gradients_and_reset(scale):
    transform = TorchAffineTransform(scale=scale, offset=1, dtype=torch.float64)
    assert isinstance(transform, AffineTransform)
    values = torch.tensor([1.0, 3.0], dtype=torch.float64, requires_grad=True)
    result = transform(values)
    torch.testing.assert_close(result, scale * values + 1)
    torch.testing.assert_close(transform.transform(values), result)
    torch.testing.assert_close(transform.untransform(result), values)
    result.sum().backward()
    torch.testing.assert_close(values.grad, torch.full_like(values, scale))
    assert transform.scale.grad.item() == 4
    assert transform.offset.grad.item() == 2
    transform.zero_grad()
    outputs = torch.tensor([3.0, 7.0], dtype=torch.float64, requires_grad=True)
    transform.untransform(outputs).sum().backward()
    torch.testing.assert_close(outputs.grad, torch.full_like(outputs, 1 / scale))
    assert transform.scale.grad.item() == -2
    assert transform.offset.grad.item() == -2 / scale
    scale_parameter = transform.scale
    offset = transform.offset
    transform.reset_to_identity()
    assert transform.scale is scale_parameter and transform.offset is offset
    torch.testing.assert_close(transform(values), values)
    assert transform(torch.tensor(1.0)).shape == torch.Size([])
    with pytest.raises(ValueError, match="finite"):
        TorchAffineTransform(scale=float("nan"))
    with pytest.raises(TypeError, match="floating-point"):
        TorchAffineTransform(dtype=torch.int64)
    with pytest.raises(ValueError, match="nonzero"):
        TorchAffineTransform(scale=0)


def test_affine_scale_can_change_sign_during_optimization():
    transform = TorchAffineTransform()
    optimizer = torch.optim.SGD([transform.scale], lr=2)
    optimizer.zero_grad()
    transform(torch.tensor(1.0)).backward()
    optimizer.step()
    assert transform.scale.item() == -1
    torch.testing.assert_close(
        transform.untransform(transform(torch.tensor(1.0))), torch.tensor(1.0)
    )


def test_torch_transform_rejects_optimizer_step_to_zero():
    transform = TorchAffineTransform()
    optimizer = torch.optim.SGD([transform.scale], lr=1)
    transform(torch.tensor(1.0)).backward()
    optimizer.step()
    assert transform.scale.item() == 0
    with pytest.raises(ValueError, match="nonzero"):
        transform(torch.tensor(1.0))
    with pytest.raises(ValueError, match="nonzero"):
        transform.untransform(torch.tensor(1.0))


def test_generic_transform_rejects_scale_changed_to_zero():
    transform = AffineTransform()
    transform.scale = 0
    with pytest.raises(ValueError, match="nonzero"):
        transform.transform(1.0)
    with pytest.raises(ValueError, match="nonzero"):
        transform.untransform(1.0)


def test_torch_registration_state_and_device_lifecycle():
    model = TensorModel()
    transform = TorchAffineTransform()
    wrapper = TorchModelWithTransforms(model, transforms={"x": transform})
    assert wrapper.transforms["x"] is transform
    assert not hasattr(model, "transforms")
    assert list(model.parameters()) == [model.curvature]
    assert set(model.state_dict()) == {"curvature"}
    assert transform.scale.dtype == torch.float64
    assert isinstance(wrapper, ModelWithTransforms)
    assert isinstance(wrapper, LUMEModel)
    assert {id(p) for p in wrapper.parameters()} == {
        id(model.curvature),
        id(transform.scale),
        id(transform.offset),
    }
    assert len(list(wrapper.parameters())) == 3
    assert {id(p) for p in wrapper.transform_parameters} == {
        id(transform.scale),
        id(transform.offset),
    }
    assert set(wrapper.state_dict()) == {
        "model.curvature",
        "transforms.x.scale",
        "transforms.x.offset",
    }
    saved = deepcopy(wrapper.state_dict())
    with torch.no_grad():
        transform.scale.fill_(2)
    wrapper.load_state_dict(saved)
    torch.testing.assert_close(transform.scale, torch.ones_like(transform.scale))
    clone = deepcopy(wrapper)
    assert clone.transforms["x"].scale is not transform.scale
    wrapper.to(dtype=torch.float32)
    assert all(p.dtype == torch.float32 for p in wrapper.parameters())
    wrapper.eval()
    assert not model.training and not transform.training
    transform.offset.requires_grad_(False)
    assert {id(p) for p in get_trainable_parameters(wrapper)} == {
        id(model.curvature),
        id(transform.scale),
    }
    assert len(wrapper.transform_parameters) == 1
    assert wrapper.transform_parameters[0] is transform.scale
    transform.scale.requires_grad_(False)
    assert wrapper.transform_parameters == []


def test_constructor_validates_before_attaching_transforms():
    model = TensorModel()
    with pytest.raises(ValueError, match="unknown transform"):
        TorchModelWithTransforms(model, {"missing": TorchAffineTransform()})
    with pytest.raises(TypeError, match="nn.Module"):
        TorchModelWithTransforms(model, {"x": AffineTransform()})
    shared = TorchAffineTransform()
    with pytest.raises(ValueError, match="independent"):
        TorchModelWithTransforms(model, {"x": shared, "z": shared})
    assert not hasattr(model, "transforms")
    wrapper = TorchModelWithTransforms(model, {"x": TorchAffineTransform()})
    second = TorchModelWithTransforms(model, {"x": TorchAffineTransform(scale=2)})
    assert wrapper.model is model
    assert second.model is model
    assert wrapper.transforms["x"] is not second.transforms["x"]
    wrapper.set_value("x", torch.tensor(1.0, dtype=torch.float64))
    assert model.x.item() == pytest.approx(1.0)
    second.set_value("x", torch.tensor(1.0, dtype=torch.float64))
    assert model.x.item() == pytest.approx(2.0)
    assert not hasattr(model, "transforms")


def test_transform_registration_validation():
    model = TensorModel()
    with pytest.raises(ValueError, match="nonempty"):
        TorchModelWithTransforms(model, {})
    with pytest.raises(ValueError, match="nonempty strings"):
        TorchModelWithTransforms(model, {"": TorchAffineTransform()})
    with pytest.raises(TypeError, match="nn.Module"):
        TorchModelWithTransforms(ScalarModel(), {"x": TorchAffineTransform()})
    with pytest.raises(TypeError, match="mapping"):
        TorchModelWithTransforms(model, None)
    with pytest.raises(TypeError, match="transform and untransform"):
        TorchModelWithTransforms(model, {"a": torch.nn.Identity()})
    with pytest.raises(KeyError):
        TorchModelWithTransforms(
            model,
            {"valid": TorchAffineTransform(), "invalid.name": TorchAffineTransform()},
        )
    assert not hasattr(model, "transforms")


def test_underlying_transform_attribute_is_not_used_or_replaced():
    model = TensorModel()
    existing = {"x": AffineTransform(10)}
    model.transforms = existing
    wrapper = TorchModelWithTransforms(model, {"x": TorchAffineTransform(scale=2)})
    wrapper.set_value("x", torch.tensor(1.0, dtype=torch.float64))
    assert model.x.item() == pytest.approx(2.0)
    assert model.transforms is existing


def test_calibration_learns_transform_and_model_parameters():
    model = TensorModel()
    wrapper = TorchModelWithTransforms(model, {"x": TorchAffineTransform()})
    x = torch.linspace(-1, 1, 11, dtype=torch.float64)
    effective = 1.15 * x + 0.1
    targets = effective + 0.3 * effective.square()
    optimizer = torch.optim.Adam(
        [{"params": [model.curvature]}, {"params": wrapper.transform_parameters}],
        lr=0.03,
    )
    history = calibrate_model(wrapper, {"x": x}, {"y": targets}, optimizer, steps=300)
    assert history[-1] < history[0] * 1e-6
    assert model.curvature.item() == pytest.approx(0.3, abs=1e-4)
    assert wrapper.transforms["x"].scale.item() == pytest.approx(1.15, abs=1e-4)
    assert wrapper.transforms["x"].offset.item() == pytest.approx(0.1, abs=1e-4)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in wrapper.parameters()
    )


def test_calibration_learns_read_only_output_transform():
    model = TensorModel()
    model.curvature.requires_grad_(False)
    wrapper = TorchModelWithTransforms(model, {"y": TorchAffineTransform()})
    x = torch.linspace(-1, 1, 11, dtype=torch.float64)
    native_targets = x + 0.2 * x.square()
    targets = (native_targets - 0.1) / 1.2
    optimizer = torch.optim.Adam(get_trainable_parameters(wrapper), lr=0.03)
    history = calibrate_model(wrapper, {"x": x}, {"y": targets}, optimizer, steps=300)
    transform = wrapper.transforms["y"]
    assert history[-1] < history[0] * 1e-6
    assert transform.scale.item() == pytest.approx(1.2, abs=1e-4)
    assert transform.offset.item() == pytest.approx(0.1, abs=1e-4)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in get_trainable_parameters(wrapper)
    )
    with pytest.raises(ReadOnlyError):
        wrapper.set_value("y", torch.tensor(1.0, dtype=torch.float64))


def test_wrapper_calls_transform_directly_in_train_and_eval_modes():
    class ExplicitTransform(TorchAffineTransform):
        def forward(self, value):
            raise AssertionError("wrapper must call transform, not forward")

    model = TensorModel()
    wrapper = TorchModelWithTransforms(
        model, {"x": ExplicitTransform(scale=2, offset=1)}
    )
    for training in (True, False):
        wrapper.train(training)
        wrapper.set({"x": torch.tensor(2.0, dtype=torch.float64)})
        torch.testing.assert_close(model.x, model.curvature.new_tensor(5.0))
