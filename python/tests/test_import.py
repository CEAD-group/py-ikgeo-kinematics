def test_extension_imports() -> None:
    import py_ikgeo_kinematics
    from py_ikgeo_kinematics import _internal

    assert set(py_ikgeo_kinematics.__all__) >= {"KinematicModel", "Robot"}
    assert _internal is not None
