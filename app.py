from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


UNPHYSICAL_THRESHOLD = 1e10
DISPLAY_COLORSCALE = [
    [0.0, "#081c15"],
    [0.2, "#1b4332"],
    [0.4, "#2d6a4f"],
    [0.6, "#52b788"],
    [0.8, "#95d5b2"],
    [1.0, "#d8f3dc"],
]
ARTIFACT_OVERLAY = [
    [0.0, "rgba(0,0,0,0)"],
    [0.5, "rgba(255,159,28,0.45)"],
    [1.0, "rgba(208,0,0,0.95)"],
]
DEMO_FILENAME = "osipi_visual_qc_demo.nii"


@dataclass(frozen=True)
class ValidationSummary:
    total_voxels: int
    nan_voxels: int
    inf_voxels: int
    unphysical_voxels: int
    corrupted_voxels: int
    corruption_ratio: float
    worst_slice_index: int
    worst_slice_corrupted_voxels: int


@dataclass(frozen=True)
class DemoCase:
    name: str
    file_bytes: bytes


def load_nifti(file_bytes: bytes, original_name: str) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    suffix = "".join(Path(original_name).suffixes).lower()
    if suffix not in {".nii", ".nii.gz"}:
        raise ValueError("Unsupported file type. Please upload a .nii or .nii.gz file.")

    try:
        loaded_image = nib.Nifti1Image.from_bytes(file_bytes)
        data = np.asarray(loaded_image.get_fdata(dtype=np.float32))
        image = nib.Nifti1Image(data, affine=loaded_image.affine, header=loaded_image.header.copy())
    except Exception as exc:  # pragma: no cover - nibabel raises several exception types
        raise ValueError("The uploaded file could not be parsed as a valid NIfTI image.") from exc

    if data.ndim not in (3, 4):
        raise ValueError("Only 3D and 4D NIfTI images are supported.")

    return image, data


def create_demo_volume(shape: tuple[int, int, int] = (72, 72, 36)) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, shape[0], dtype=np.float32)
    y = np.linspace(-1.0, 1.0, shape[1], dtype=np.float32)
    z = np.linspace(-1.0, 1.0, shape[2], dtype=np.float32)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing="ij")

    ellipsoid = np.exp(-((grid_x * 1.2) ** 2 + (grid_y * 0.95) ** 2 + (grid_z * 1.6) ** 2) * 4.0)
    cortex = 1250 * ellipsoid
    ventricles = 320 * np.exp(-(((grid_x * 2.1) ** 2) + ((grid_y * 2.6) ** 2) + ((grid_z * 2.0) ** 2)) * 5.5)
    texture = 50 * np.sin(8 * grid_x) * np.cos(7 * grid_y) * np.exp(-(grid_z**2) * 2.5)
    volume = cortex - ventricles + texture
    volume = np.maximum(volume, 0).astype(np.float32)

    worst_slice = shape[2] // 2 + 2
    volume[20:26, 24:31, worst_slice] = np.nan
    volume[43:49, 34:42, worst_slice] = np.inf
    volume[30:36, 46:52, worst_slice] = UNPHYSICAL_THRESHOLD * 25
    volume[34:38, 19:27, worst_slice - 1] = UNPHYSICAL_THRESHOLD * 5
    return volume


def create_demo_case() -> DemoCase:
    image = nib.Nifti1Image(create_demo_volume(), affine=np.eye(4))
    return DemoCase(name=DEMO_FILENAME, file_bytes=image.to_bytes())


def detect_artifacts(data: np.ndarray, threshold: float = UNPHYSICAL_THRESHOLD) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nan_mask = np.isnan(data)
    inf_mask = np.isinf(data)
    finite_values = np.where(np.isfinite(data), np.abs(data), 0)
    unphysical_mask = finite_values > threshold
    combined_mask = nan_mask | inf_mask | unphysical_mask
    return nan_mask, inf_mask, unphysical_mask, combined_mask


def choose_volume(data: np.ndarray, volume_index: int | None = None) -> tuple[np.ndarray, int | None]:
    if data.ndim == 3:
        return data, None

    volume_count = data.shape[3]
    if volume_index is None:
        artifact_totals = []
        for idx in range(volume_count):
            _, _, _, combined_mask = detect_artifacts(data[..., idx])
            artifact_totals.append(int(combined_mask.sum()))
        volume_index = int(np.argmax(artifact_totals))

    if not 0 <= volume_index < volume_count:
        raise ValueError(f"Volume index must be between 0 and {volume_count - 1}.")

    return data[..., volume_index], volume_index


def summarize_volume(volume: np.ndarray, threshold: float = UNPHYSICAL_THRESHOLD) -> ValidationSummary:
    nan_mask, inf_mask, unphysical_mask, combined_mask = detect_artifacts(volume, threshold=threshold)
    slice_totals = combined_mask.sum(axis=(0, 1))
    worst_slice_index = int(np.argmax(slice_totals))
    corrupted_voxels = int(combined_mask.sum())
    total_voxels = int(volume.size)

    return ValidationSummary(
        total_voxels=total_voxels,
        nan_voxels=int(nan_mask.sum()),
        inf_voxels=int(inf_mask.sum()),
        unphysical_voxels=int(unphysical_mask.sum()),
        corrupted_voxels=corrupted_voxels,
        corruption_ratio=(corrupted_voxels / total_voxels) if total_voxels else 0.0,
        worst_slice_index=worst_slice_index,
        worst_slice_corrupted_voxels=int(slice_totals[worst_slice_index]),
    )


def sanitize_data(data: np.ndarray, threshold: float = UNPHYSICAL_THRESHOLD, fill_value: float = 0.0) -> np.ndarray:
    sanitized = np.array(data, copy=True)
    _, _, _, combined_mask = detect_artifacts(sanitized, threshold=threshold)
    sanitized[combined_mask] = fill_value
    return sanitized


def extract_slice(volume: np.ndarray, slice_index: int) -> np.ndarray:
    return np.rot90(volume[:, :, slice_index])


def intensity_window(slice_2d: np.ndarray) -> tuple[float, float]:
    finite_values = slice_2d[np.isfinite(slice_2d)]
    if finite_values.size == 0:
        return 0.0, 1.0

    lower = float(np.percentile(finite_values, 1))
    upper = float(np.percentile(finite_values, 99))
    if np.isclose(lower, upper):
        upper = lower + 1.0
    return lower, upper


def build_slice_figure(
    corrupted_slice: np.ndarray,
    sanitized_slice: np.ndarray,
    artifact_slice_mask: np.ndarray,
    slice_index: int,
    volume_index: int | None,
) -> go.Figure:
    zmin, zmax = intensity_window(sanitized_slice)
    subtitle = f"Axial Slice z={slice_index}"
    if volume_index is not None:
        subtitle += f" | Volume t={volume_index}"

    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Corrupted Slice", "Sanitized Slice"),
        horizontal_spacing=0.08,
    )
    figure.add_trace(
        go.Heatmap(
            z=corrupted_slice,
            colorscale=DISPLAY_COLORSCALE,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title="Intensity", len=0.8, thickness=16),
            hovertemplate="x=%{x}<br>y=%{y}<br>value=%{z:.4g}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Heatmap(
            z=artifact_slice_mask.astype(int),
            colorscale=ARTIFACT_OVERLAY,
            showscale=False,
            hovertemplate="Artifact mask: %{z}<extra></extra>",
            opacity=0.9,
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Heatmap(
            z=sanitized_slice,
            colorscale=DISPLAY_COLORSCALE,
            zmin=zmin,
            zmax=zmax,
            showscale=False,
            hovertemplate="x=%{x}<br>y=%{y}<br>value=%{z:.4g}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    figure.update_xaxes(showticklabels=False)
    figure.update_yaxes(showticklabels=False, scaleanchor="x", scaleratio=1)
    figure.update_annotations(font=dict(size=14, color="#f3fff9"), yshift=8)
    figure.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=110, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(3,15,13,1)",
        font=dict(color="#e8fff6", family="Segoe UI, Arial, sans-serif"),
        title=dict(
            text=f"{subtitle}<br><sup>Artifact hotspots are highlighted in amber/red.</sup>",
            x=0.5,
            y=0.94,
            xanchor="center",
            yanchor="top",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return figure


def build_download_payload(image: nib.spatialimages.SpatialImage, sanitized_data: np.ndarray, original_name: str) -> tuple[bytes, str]:
    new_header = image.header.copy()
    sanitized_image = nib.Nifti1Image(sanitized_data.astype(np.float32), affine=image.affine, header=new_header)

    suffix = "".join(Path(original_name).suffixes).lower() or ".nii.gz"
    if original_name.lower().endswith(".nii.gz"):
        output_name = f"{original_name[:-7]}_sanitized.nii.gz"
    elif original_name.lower().endswith(".nii"):
        output_name = f"{original_name[:-4]}_sanitized.nii"
    else:
        output_name = f"sanitized_output{suffix}"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        temp_path = Path(handle.name)

    try:
        nib.save(sanitized_image, str(temp_path))
        payload = temp_path.read_bytes()
    finally:
        temp_path.unlink(missing_ok=True)

    return payload, output_name


def format_large_number(value: int) -> str:
    return f"{value:,}"


def render_header() -> None:
    st.set_page_config(page_title="OSIPI Visual QC", page_icon="🧠", layout="wide")
    st.markdown(
        """
        <style>
            [data-testid="stHeader"],
            [data-testid="stToolbar"],
            [data-testid="stDecoration"],
            .stDeployButton {
                display: none;
            }
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(36, 196, 255, 0.18), transparent 28%),
                    radial-gradient(circle at 85% 12%, rgba(255, 159, 28, 0.22), transparent 26%),
                    radial-gradient(circle at 50% 100%, rgba(76, 201, 240, 0.10), transparent 34%),
                    linear-gradient(160deg, #051311 0%, #0b201c 52%, #102219 100%);
                color: #f3fff9;
            }
            .block-container {
                padding-top: 1.4rem;
                padding-bottom: 2rem;
                max-width: 1180px;
            }
            .hero-card, .metric-card, .workflow-card, .empty-state-card, .source-card, .summary-shell {
                background: linear-gradient(180deg, rgba(7, 27, 23, 0.82), rgba(5, 22, 18, 0.72));
                border: 1px solid rgba(124, 241, 214, 0.16);
                box-shadow: 0 24px 60px rgba(0, 0, 0, 0.26);
                backdrop-filter: blur(16px);
                border-radius: 24px;
            }
            .hero-card {
                padding: 1.6rem 1.8rem;
                margin-bottom: 1rem;
            }
            .metric-card {
                padding: 1rem 1.2rem;
                min-height: 128px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            }
            .workflow-card, .empty-state-card, .source-card, .summary-shell {
                padding: 1.15rem 1.2rem;
            }
            .source-card {
                margin-bottom: 0.85rem;
            }
            .workflow-card {
                margin-bottom: 1rem;
            }
            .summary-shell {
                min-height: 214px;
            }
            .eyebrow {
                color: #95d5b2;
                text-transform: uppercase;
                letter-spacing: 0.16em;
                font-size: 0.8rem;
                font-weight: 700;
            }
            .hero-title {
                margin: 0.35rem 0 0.6rem;
                font-size: 2.7rem;
                line-height: 1.05;
                font-weight: 800;
            }
            .hero-copy {
                margin: 0;
                color: #d8f3dc;
                font-size: 1.05rem;
                max-width: 52rem;
            }
            .section-title {
                margin: 0 0 0.25rem;
                font-size: 1.15rem;
                font-weight: 700;
                color: #f3fff9;
            }
            .summary-title {
                margin: 0 0 0.35rem;
                font-size: 2rem;
                line-height: 1.05;
                font-weight: 800;
                color: #f5fffb;
            }
            .section-kicker {
                margin: 0 0 0.75rem;
                color: #9ae6d1;
                font-size: 0.95rem;
            }
            .summary-metric-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 0.85rem;
                margin-top: 1rem;
            }
            .summary-metric-tile {
                background: linear-gradient(180deg, rgba(7, 26, 22, 0.82), rgba(5, 21, 18, 0.74));
                border: 1px solid rgba(124, 241, 214, 0.12);
                border-radius: 18px;
                padding: 0.95rem 0.9rem;
                min-height: 112px;
            }
            .summary-metric-label {
                color: #d7fff2;
                font-size: 0.92rem;
                font-weight: 600;
                margin-bottom: 0.55rem;
                word-break: keep-all;
                overflow-wrap: normal;
                white-space: nowrap;
            }
            .summary-metric-value {
                color: #f7fffb;
                font-size: 2.1rem;
                font-weight: 800;
                line-height: 1.05;
                margin: 0;
            }
            .summary-footnote {
                margin: 0.95rem 0 0;
                color: #ccebdd;
                font-size: 0.95rem;
            }
            .section-copy {
                margin: 0 0 0.9rem;
                color: #ccebdd;
            }
            .workflow-list {
                margin: 0.5rem 0 0;
                padding-left: 1.1rem;
                color: #eafaf3;
                line-height: 1.75;
            }
            .workflow-list li {
                margin-bottom: 0.35rem;
            }
            .source-meta {
                margin: 0.35rem 0 0;
                color: #d8f3dc;
            }
            .empty-state-title {
                margin: 0.25rem 0 0.4rem;
                font-size: 1.25rem;
                font-weight: 700;
            }
            .empty-state-copy {
                margin: 0;
                color: #ccebdd;
                max-width: 45rem;
            }
            .mini-label {
                color: #95d5b2;
                font-size: 0.82rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                margin-bottom: 0.3rem;
            }
            .mini-value {
                font-size: 1.8rem;
                font-weight: 800;
                color: #f4fff8;
                margin: 0;
            }
            .mini-copy {
                color: #ccebdd;
                margin: 0.3rem 0 0;
                flex-grow: 1;
            }
            [data-testid="stFileUploaderDropzone"] {
                background: linear-gradient(180deg, rgba(6, 17, 15, 0.92), rgba(9, 28, 23, 0.85));
                border: 1px dashed rgba(124, 241, 214, 0.45);
                border-radius: 22px;
                padding: 1.05rem;
                min-height: 148px;
                box-shadow: inset 0 0 0 1px rgba(76, 201, 240, 0.05);
            }
            [data-testid="stFileUploaderDropzone"] * {
                color: #f3fff9 !important;
            }
            [data-testid="stFileUploaderDropzoneInstructions"] span {
                color: #ccebdd !important;
            }
            [data-testid="stBaseButton-secondary"],
            [data-testid="stBaseButton-primary"] {
                border-radius: 999px;
            }
            .stButton button, .stDownloadButton button {
                min-height: 2.9rem;
                font-weight: 700;
                border: 1px solid rgba(124, 241, 214, 0.18);
                background: linear-gradient(180deg, rgba(20, 29, 45, 0.95), rgba(18, 24, 38, 0.95));
                color: #f5fffb;
                box-shadow: 0 10px 24px rgba(0, 0, 0, 0.18);
            }
            .stDownloadButton button[kind="primary"] {
                background: linear-gradient(135deg, #1b9aaa, #27c2a5);
                color: #041512;
            }
            [data-testid="stMetric"] {
                background: linear-gradient(180deg, rgba(7, 26, 22, 0.82), rgba(5, 21, 18, 0.74));
                border: 1px solid rgba(124, 241, 214, 0.12);
                padding: 0.8rem;
                border-radius: 18px;
            }
            [data-testid="stAlert"] {
                border-radius: 20px;
                margin-bottom: 0.9rem;
            }
            .stSlider {
                padding-top: 0.25rem;
            }
            .export-shell {
                margin-top: 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero-card">
            <div class="eyebrow">OSIPI Challenge Support</div>
            <h1 class="hero-title">Visual QC for MRI Challenge Submissions</h1>
            <p class="hero-copy">
                Upload a NIfTI submission and instantly inspect corruption hotspots, quantify invalid voxels,
                and export a sanitized image that is safe for downstream scoring pipelines.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="mini-label">{label}</div>
            <p class="mini-value">{value}</p>
            <p class="mini-copy">{copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workflow_card() -> None:
    st.markdown(
        """
        <div class="workflow-card">
            <div class="eyebrow">Review Workflow</div>
            <p class="section-title">Fast triage for challenge submissions</p>
            <ol class="workflow-list">
                <li>Upload a 3D or 4D NIfTI submission or launch the built-in demo scan.</li>
                <li>Review the most corrupted axial slice with interactive overlays and metrics.</li>
                <li>Download a sanitized output that is ready for downstream scoring.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty-state-card">
            <div class="eyebrow">Ready When You Are</div>
            <p class="empty-state-title">Load a scan to activate the QC dashboard</p>
            <p class="empty-state-copy">
                Use your own `.nii` or `.nii.gz` file, or tap the demo button to preview the full experience with a synthetic MRI volume that contains realistic artifact hotspots.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_source_card(source_name: str, source_kind: str) -> None:
    st.markdown(
        f"""
        <div class="source-card">
            <div class="eyebrow">Current Source</div>
            <p class="section-title">{source_name}</p>
            <p class="source-meta">Loaded from {source_kind}.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_panel(summary: ValidationSummary, image_shape: tuple[int, ...]) -> None:
    if summary.corrupted_voxels == 0:
        st.success("Validation Passed! No NaN, Inf, or unphysical intensities were detected in the selected volume.")
    else:
        st.error("Artifacts Detected! Review the highlighted slice and export the sanitized image for downstream use.")

    render_metric_card(
        "Image Shape",
        " x ".join(str(dimension) for dimension in image_shape),
        "3D and 4D NIfTI images are supported.",
    )


def render_stats_panel(summary: ValidationSummary) -> None:
    st.markdown(
        f"""
        <div class="summary-shell">
            <div class="eyebrow">Quantitative Analysis</div>
            <p class="summary-title">QC Summary</p>
            <p class="section-kicker">Instant quantitative triage for the currently selected scan.</p>
            <div class="summary-metric-grid">
                <div class="summary-metric-tile">
                    <div class="summary-metric-label">Total Voxels</div>
                    <p class="summary-metric-value">{format_large_number(summary.total_voxels)}</p>
                </div>
                <div class="summary-metric-tile">
                    <div class="summary-metric-label">Corrupted</div>
                    <p class="summary-metric-value">{format_large_number(summary.corrupted_voxels)}</p>
                </div>
                <div class="summary-metric-tile">
                    <div class="summary-metric-label">NaN + Inf</div>
                    <p class="summary-metric-value">{format_large_number(summary.nan_voxels + summary.inf_voxels)}</p>
                </div>
                <div class="summary-metric-tile">
                    <div class="summary-metric-label">Unphysical</div>
                    <p class="summary-metric-value">{format_large_number(summary.unphysical_voxels)}</p>
                </div>
            </div>
            <p class="summary-footnote">
                Corruption ratio: {summary.corruption_ratio:.3%} | Worst slice: z={summary.worst_slice_index} with {summary.worst_slice_corrupted_voxels:,} corrupted voxels.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard() -> None:
    render_header()

    if "demo_case" not in st.session_state:
        st.session_state.demo_case = None

    left_column, right_column = st.columns([1, 1], gap="large")

    with left_column:
        st.markdown("#### Input")
        st.markdown(
            '<p class="section-copy">Upload a contestant submission or launch the built-in demo scan to preview the complete workflow.</p>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Upload Contestant NIfTI Submission",
            type=["nii", "gz"],
            help="Accepted formats: .nii and .nii.gz",
            label_visibility="collapsed",
        )
        button_col_1, button_col_2 = st.columns(2, gap="small")
        with button_col_1:
            if st.button("Load Demo Scan", use_container_width=True, key="load_demo"):
                st.session_state.demo_case = create_demo_case()
        with button_col_2:
            if st.button(
                "Clear Demo",
                use_container_width=True,
                key="clear_demo",
                disabled=st.session_state.demo_case is None,
            ):
                st.session_state.demo_case = None
        st.caption("Designed for fast visual triage by clinical researchers and challenge organizers.")

    with right_column:
        render_workflow_card()
        capability_col_1, capability_col_2 = st.columns(2, gap="medium")
        with capability_col_1:
            render_metric_card("Checks", "NaN / Inf / >1e10", "Deterministic artifact screening.")
        with capability_col_2:
            render_metric_card("Repair", "Zero-fill", "Stable, reproducible sanitization default.")

    selected_name: str | None = None
    selected_bytes: bytes | None = None
    selected_source: str | None = None

    if uploaded_file is not None:
        selected_name = uploaded_file.name
        selected_bytes = uploaded_file.getvalue()
        selected_source = "upload"
    elif st.session_state.demo_case is not None:
        demo_case: DemoCase = st.session_state.demo_case
        selected_name = demo_case.name
        selected_bytes = demo_case.file_bytes
        selected_source = "demo data"

    if selected_bytes is None or selected_name is None or selected_source is None:
        render_empty_state()
        return

    try:
        image, data = load_nifti(selected_bytes, selected_name)
    except ValueError as exc:
        st.error(str(exc))
        return

    if data.ndim == 4:
        _, default_index = choose_volume(data)
        st.markdown("### Volume Selection")
        chosen_volume_index = st.slider(
            "4D image detected. Select the timepoint to inspect.",
            min_value=0,
            max_value=data.shape[3] - 1,
            value=default_index or 0,
        )
        volume, volume_index = choose_volume(data, volume_index=chosen_volume_index)
    else:
        volume, volume_index = choose_volume(data)

    summary = summarize_volume(volume)
    sanitized_volume = sanitize_data(volume)
    sanitized_full = sanitize_data(data)
    _, _, _, volume_mask = detect_artifacts(volume)

    summary_col, stats_col = st.columns([1, 1], gap="large")
    with summary_col:
        render_source_card(selected_name, selected_source)
        render_status_panel(summary, data.shape)

    with stats_col:
        render_stats_panel(summary)

    corrupted_slice = extract_slice(volume, summary.worst_slice_index)
    sanitized_slice = extract_slice(sanitized_volume, summary.worst_slice_index)
    artifact_slice_mask = np.rot90(volume_mask[:, :, summary.worst_slice_index])

    st.markdown("### Corruption Hotspot Viewer")
    figure = build_slice_figure(
        corrupted_slice=corrupted_slice,
        sanitized_slice=sanitized_slice,
        artifact_slice_mask=artifact_slice_mask,
        slice_index=summary.worst_slice_index,
        volume_index=volume_index,
    )
    st.plotly_chart(figure, use_container_width=True)

    insight_col_1, insight_col_2, insight_col_3 = st.columns(3, gap="medium")
    with insight_col_1:
        render_metric_card("Worst Slice", f"z = {summary.worst_slice_index}", "Maximum artifact concentration.")
    with insight_col_2:
        render_metric_card("Slice Burden", format_large_number(summary.worst_slice_corrupted_voxels), "Corrupted voxels in that slice.")
    with insight_col_3:
        render_metric_card("Sanitization", "Ready", "Full-volume corrected image prepared for export.")

    download_bytes, output_name = build_download_payload(image, sanitized_full, selected_name)
    st.markdown('<div class="export-shell">', unsafe_allow_html=True)
    st.download_button(
        "Sanitize and Export",
        data=download_bytes,
        file_name=output_name,
        mime="application/gzip" if output_name.endswith(".gz") else "application/octet-stream",
        use_container_width=True,
        type="primary",
    )
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    render_dashboard()
