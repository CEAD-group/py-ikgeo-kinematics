from typing import Any

import numpy as np
import pytest
from py_ikgeo_kinematics import KinematicModel


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("field", ["h", "p", "offsets"])
def test_nonfinite_model_values_are_rejected(field: str, value: float) -> None:
    model = KinematicModel.from_opw(c2=300, c3=250)
    h, p, offsets = model.h.copy(), model.p.copy(), np.zeros(6)
    {"h": h, "p": p, "offsets": offsets}[field].flat[0] = value
    with pytest.raises(ValueError, match="finite"):
        KinematicModel(
            h, p, offsets=(offsets[0], offsets[1], offsets[2], offsets[3], offsets[4], offsets[5])
        )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("field", ["a1", "a2", "b", "c1", "c2", "c3", "c4", "offsets"])
def test_nonfinite_opw_values_are_rejected(field: str, value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        if field == "offsets":
            KinematicModel.from_opw(offsets=(value, 0, 0, 0, 0, 0))
        else:
            parameters: dict[str, Any] = {field: value}
            KinematicModel.from_opw(**parameters)


@pytest.mark.parametrize("field", ["h", "p"])
def test_geometry_arrays_are_readonly(field: str) -> None:
    model = KinematicModel.from_opw(c2=300, c3=250)
    array = getattr(model, field)
    assert not array.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        array.flat[0] = 1
    copy = array.copy()
    copy.flat[0] = 123
    np.testing.assert_array_equal(getattr(model, field), array)
