from __future__ import annotations

from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np
from streamlit.testing.v1 import AppTest

from app import (
    DEMO_FILENAME,
    UNPHYSICAL_THRESHOLD,
    build_download_payload,
    build_slice_figure,
    choose_volume,
    create_demo_case,
    create_demo_volume,
    detect_artifacts,
    extract_slice,
    intensity_window,
    load_nifti,
    sanitize_data,
    summarize_volume,
)


def make_nifti_bytes(array: np.ndarray) -> bytes:
    image = nib.Nifti1Image(array.astype(np.float32), affine=np.eye(4))
    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as handle:
        temp_path = Path(handle.name)

    try:
        nib.save(image, str(temp_path))
        return temp_path.read_bytes()
    finally:
        temp_path.unlink(missing_ok=True)


def test_detect_artifacts_counts_nan_inf_and_unphysical_values() -> None:
    data = np.array(
        [
            [[0.0, np.nan], [np.inf, 5.0]],
            [[UNPHYSICAL_THRESHOLD * 2, -3.0], [7.0, 8.0]],
        ],
        dtype=np.float32,
    )

    nan_mask, inf_mask, unphysical_mask, combined_mask = detect_artifacts(data)

    assert int(nan_mask.sum()) == 1
    assert int(inf_mask.sum()) == 1
    assert int(unphysical_mask.sum()) == 1
    assert int(combined_mask.sum()) == 3


def test_choose_volume_defaults_to_the_most_corrupted_timepoint() -> None:
    data = np.zeros((2, 2, 2, 3), dtype=np.float32)
    data[0, 0, 0, 1] = np.nan
    data[0, 1, 0, 1] = np.inf
    data[1, 1, 1, 2] = np.nan

    volume, index = choose_volume(data)

    assert index == 1
    np.testing.assert_array_equal(volume, data[..., 1])


def test_summarize_volume_reports_worst_slice_and_ratio() -> None:
    volume = np.zeros((3, 3, 4), dtype=np.float32)
    volume[0, 0, 2] = np.nan
    volume[1, 1, 2] = np.inf
    volume[2, 2, 3] = UNPHYSICAL_THRESHOLD * 5

    summary = summarize_volume(volume)

    assert summary.total_voxels == 36
    assert summary.corrupted_voxels == 3
    assert summary.worst_slice_index == 2
    assert summary.worst_slice_corrupted_voxels == 2
    assert np.isclose(summary.corruption_ratio, 3 / 36)


def test_sanitize_data_replaces_all_detected_artifacts_with_zero() -> None:
    volume = np.array([[[np.nan, np.inf, UNPHYSICAL_THRESHOLD * 2, 4.0]]], dtype=np.float32)

    sanitized = sanitize_data(volume)

    np.testing.assert_array_equal(sanitized, np.array([[[0.0, 0.0, 0.0, 4.0]]], dtype=np.float32))


def test_load_nifti_supports_valid_3d_images() -> None:
    original = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    file_bytes = make_nifti_bytes(original)

    image, loaded = load_nifti(file_bytes, "sample.nii")

    assert image.shape == (3, 3, 3)
    np.testing.assert_allclose(loaded, original)


def test_load_nifti_rejects_invalid_extension() -> None:
    try:
        load_nifti(b"not-a-nifti", "sample.txt")
    except ValueError as exc:
        assert "Unsupported file type" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid extension")


def test_choose_volume_raises_for_out_of_range_index() -> None:
    data = np.zeros((2, 2, 2, 2), dtype=np.float32)

    try:
        choose_volume(data, volume_index=5)
    except ValueError as exc:
        assert "Volume index must be between 0 and 1" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid volume index")


def test_create_demo_volume_contains_deterministic_artifacts() -> None:
    volume = create_demo_volume()
    summary = summarize_volume(volume)

    assert volume.shape == (72, 72, 36)
    assert summary.corrupted_voxels > 0
    assert summary.worst_slice_index == 20
    assert summary.worst_slice_corrupted_voxels > 0


def test_summarize_volume_reports_clean_scan() -> None:
    volume = np.ones((4, 4, 4), dtype=np.float32)

    summary = summarize_volume(volume)

    assert summary.corrupted_voxels == 0
    assert summary.nan_voxels == 0
    assert summary.inf_voxels == 0
    assert summary.unphysical_voxels == 0
    assert summary.corruption_ratio == 0.0


def test_intensity_window_handles_constant_slice() -> None:
    slice_2d = np.full((4, 4), 5.0, dtype=np.float32)

    zmin, zmax = intensity_window(slice_2d)

    assert zmin == 5.0
    assert zmax == 6.0


def test_build_slice_figure_contains_three_heatmaps() -> None:
    volume = create_demo_volume()
    summary = summarize_volume(volume)
    sanitized = sanitize_data(volume)
    _, _, _, mask = detect_artifacts(volume)

    corrupted_slice = extract_slice(volume, summary.worst_slice_index)
    sanitized_slice = extract_slice(sanitized, summary.worst_slice_index)
    artifact_slice_mask = np.rot90(mask[:, :, summary.worst_slice_index])

    figure = build_slice_figure(
        corrupted_slice=corrupted_slice,
        sanitized_slice=sanitized_slice,
        artifact_slice_mask=artifact_slice_mask,
        slice_index=summary.worst_slice_index,
        volume_index=None,
    )

    assert len(figure.data) == 3
    assert figure.layout.title.text is not None
    assert "Axial Slice" in figure.layout.title.text


def test_app_smoke_renders_demo_case_dashboard() -> None:
    at = AppTest.from_file("app.py")
    demo_case = create_demo_case()
    at.session_state["demo_case"] = demo_case

    at.run()

    assert len(at.exception) == 0
    assert len(at.error) == 1
    assert "Artifacts Detected!" in at.error[0].value
    assert any("Current Source" in markdown.value for markdown in at.markdown)
    assert any("QC Summary" in markdown.value for markdown in at.markdown)


def test_build_download_payload_preserves_shape_and_replaces_filename() -> None:
    array = np.zeros((2, 2, 2), dtype=np.float32)
    image = nib.Nifti1Image(array, affine=np.eye(4))
    sanitized = np.ones((2, 2, 2), dtype=np.float32)

    payload, output_name = build_download_payload(image, sanitized, "brain.nii.gz")

    assert output_name == "brain_sanitized.nii.gz"
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as handle:
        temp_path = Path(handle.name)

    try:
        temp_path.write_bytes(payload)
        reloaded = nib.load(str(temp_path))
        np.testing.assert_array_equal(reloaded.get_fdata(dtype=np.float32), sanitized)
    finally:
        temp_path.unlink(missing_ok=True)


def test_build_download_payload_uses_demo_filename_safely() -> None:
    array = np.zeros((2, 2, 2), dtype=np.float32)
    image = nib.Nifti1Image(array, affine=np.eye(4))

    _, output_name = build_download_payload(image, array, DEMO_FILENAME)

    assert output_name == "osipi_visual_qc_demo_sanitized.nii"


def test_app_empty_state_renders_without_data() -> None:
    at = AppTest.from_file("app.py")

    at.run()

    assert len(at.exception) == 0
    assert any("Load a scan to activate the QC dashboard" in markdown.value for markdown in at.markdown)
