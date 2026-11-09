from __future__ import annotations

from pathlib import Path
from typing import Any

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# APP CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = ROOT_DIR / "assets"
DATA_DIR = ROOT_DIR / "data"

EMPLOYEE_DATA_PATH = ROOT_DIR / "hasil_cluster.csv"
CLUSTER_PROFILE_PATH = ROOT_DIR / "profil_cluster.csv"
EVALUATION_PATH = ROOT_DIR / "evaluasi_k.csv"
MODEL_PATH = ROOT_DIR / "kmeans_model.pkl"
SCALER_PATH = ROOT_DIR / "scaler.pkl"
MODEL_INFO_PATH = ROOT_DIR / "model_info.pkl"

st.set_page_config(
    page_title="Kelompok AHuyy",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

SEGMENT_COLORS = {
    "Junior": "#3478F6",
    "Baru Bergabung - Berpengalaman": "#17B6D2",
    "Menengah": "#12B886",
    "Senior": "#8B72E8",
}

REQUIRED_EMPLOYEE_COLUMNS = {
    "EmployeeNumber",
    "Department",
    "JobRole",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "TotalWorkingYears",
    "cluster",
    "segmen",
    "PC1",
    "PC2",
}


# ============================================================
# STYLING
# ============================================================

def load_stylesheet() -> None:
    """Load custom CSS from assets/style.css."""
    css_path = ASSETS_DIR / "style.css"

    if not css_path.exists():
        st.warning("File assets/style.css tidak ditemukan. Tampilan default digunakan.")
        return

    css = css_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


load_stylesheet()


# ============================================================
# DATA ACCESS
# ============================================================

@st.cache_data(show_spinner="Memuat dataset karyawan...")
def load_employee_data() -> pd.DataFrame:
    if not EMPLOYEE_DATA_PATH.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {EMPLOYEE_DATA_PATH.name}")

    df = pd.read_csv(EMPLOYEE_DATA_PATH)
    missing = REQUIRED_EMPLOYEE_COLUMNS.difference(df.columns)

    if missing:
        raise ValueError(
            "Kolom wajib tidak tersedia pada hasil_cluster.csv: "
            + ", ".join(sorted(missing))
        )

    numeric_columns = [
        "EmployeeNumber",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "TotalWorkingYears",
        "cluster",
        "PC1",
        "PC2",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=numeric_columns).copy()
    df["cluster"] = df["cluster"].astype(int)
    df["EmployeeNumber"] = df["EmployeeNumber"].astype(int)
    df["segmen"] = df["segmen"].astype(str)

    return df


@st.cache_data(show_spinner=False)
def load_cluster_profiles() -> pd.DataFrame:
    if not CLUSTER_PROFILE_PATH.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {CLUSTER_PROFILE_PATH.name}")

    profiles = pd.read_csv(CLUSTER_PROFILE_PATH)
    required = {
        "cluster",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "TotalWorkingYears",
        "jumlah",
        "persen",
        "segmen",
    }
    missing = required.difference(profiles.columns)

    if missing:
        raise ValueError(
            "Kolom wajib tidak tersedia pada profil_cluster.csv: "
            + ", ".join(sorted(missing))
        )

    return profiles.sort_values("cluster").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_evaluation() -> pd.DataFrame:
    if not EVALUATION_PATH.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {EVALUATION_PATH.name}")

    evaluation = pd.read_csv(EVALUATION_PATH)
    required = {"k", "inertia", "silhouette", "davies_bouldin"}
    missing = required.difference(evaluation.columns)

    if missing:
        raise ValueError(
            "Kolom wajib tidak tersedia pada evaluasi_k.csv: "
            + ", ".join(sorted(missing))
        )

    return evaluation.sort_values("k").reset_index(drop=True)


@st.cache_resource(show_spinner="Memuat model machine learning...")
def load_model_assets() -> tuple[Any, Any, dict[str, Any]]:
    for path in (MODEL_PATH, SCALER_PATH, MODEL_INFO_PATH):
        if not path.exists():
            raise FileNotFoundError(f"File model tidak ditemukan: {path.name}")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    model_info = joblib.load(MODEL_INFO_PATH)

    features = model_info.get("fitur", [])
    if not features:
        raise ValueError("Daftar fitur tidak ditemukan di model_info.pkl.")

    if getattr(model, "n_features_in_", len(features)) != len(features):
        raise ValueError("Jumlah fitur model tidak sesuai dengan model_info.pkl.")

    if getattr(scaler, "n_features_in_", len(features)) != len(features):
        raise ValueError("Jumlah fitur scaler tidak sesuai dengan model_info.pkl.")

    return model, scaler, model_info


# ============================================================
# SHARED UI COMPONENTS
# ============================================================

def render_topbar() -> str:
    """Render topbar and return the global search query."""
    left, right = st.columns([5, 1.3], vertical_alignment="center")

    with left:
        search_query = st.text_input(
            "Pencarian global",
            placeholder="Search employee ID, department, job role...",
            label_visibility="collapsed",
            key="global_search",
        )

    with right:
        st.markdown(
            '<div class="topbar-status"><span class="status-dot"></span> HR ANALYTICS</div>',
            unsafe_allow_html=True,
        )

    return search_query.strip()


def render_page_heading(eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="page-heading">
            <div class="page-eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(
    label: str,
    value: str,
    detail: str,
    icon: str,
    accent: str = "blue",
) -> None:
    st.markdown(
        f"""
        <div class="metric-card accent-{accent}">
            <div class="metric-card-top">
                <span class="metric-label">{label}</span>
                <span class="metric-icon">{icon}</span>
            </div>
            <div class="metric-value">{value}</div>
            <div class="metric-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_heading(title: str, subtitle: str | None = None) -> None:
    detail = f'<p>{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<div class="section-heading"><h3>{title}</h3>{detail}</div>',
        unsafe_allow_html=True,
    )


def format_number(value: float, decimals: int = 0) -> str:
    return f"{value:,.{decimals}f}".replace(",", ".")


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            """
            <div class="brand-lockup">
                <div class="brand-mark">A</div>
                <div class="brand-name">Kelompok<span>Ahuy</span></div>
            </div>
            <div class="sidebar-caption">HUMAN RESOURCES PLATFORM</div>
            <div class="sidebar-divider"></div>
            <div class="sidebar-section-label">WORKSPACE</div>
            """,
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigasi utama",
            [
                "Dashboard",
                "Employee Directory",
                "Cluster Analytics",
                "Predict Employee",
                "Model Information",
            ],
            label_visibility="collapsed",
            key="main_navigation",
        )

        st.markdown(
            """
            <div class="sidebar-bottom">
                <div class="sidebar-divider"></div>
                <div class="sidebar-section-label">PROJECT</div>
                <div class="sidebar-project-card">
                    <div class="project-card-icon">ML</div>
                    <div>
                        <strong>Employee Clustering</strong>
                        <span>K-Means · 4 Segments</span>
                    </div>
                </div>
                <div class="sidebar-version">STREAMLIT APPLICATION · V1.0</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return page


# ============================================================
# DASHBOARD
# ============================================================

def render_dashboard(
    employees: pd.DataFrame,
    profiles: pd.DataFrame,
    search_query: str,
) -> None:
    render_page_heading(
        "OVERVIEW / HUMAN RESOURCES",
        "Employee Dashboard",
        "Ringkasan segmentasi karyawan berdasarkan masa kerja dan pengalaman profesional.",
    )

    total_employees = len(employees)
    avg_tenure = employees["YearsAtCompany"].mean()
    departments = employees["Department"].nunique()
    number_of_clusters = employees["cluster"].nunique()

    metric_columns = st.columns(4, gap="medium")
    metrics = [
        ("Total Employees", format_number(total_employees), "Employees in dataset", "♙", "blue"),
        ("Average Tenure", f"{avg_tenure:.1f} yrs", "Average years at company", "◷", "cyan"),
        ("Departments", str(departments), "Departments represented", "▦", "green"),
        ("Employee Segments", str(number_of_clusters), "K-Means cluster groups", "◈", "purple"),
    ]

    for column, metric in zip(metric_columns, metrics):
        with column:
            render_metric_card(*metric)

    st.write("")

    chart_left, chart_right = st.columns([1.05, 1.45], gap="large")

    with chart_left:
        with st.container(border=True):
            render_section_heading(
                "Employee by Segment",
                "Jumlah karyawan pada setiap hasil cluster.",
            )

            segment_chart = (
                alt.Chart(profiles)
                .mark_bar(cornerRadiusEnd=5, height=24)
                .encode(
                    x=alt.X("jumlah:Q", title="Employees", axis=alt.Axis(grid=False)),
                    y=alt.Y("segmen:N", sort="-x", title=None),
                    color=alt.Color(
                        "segmen:N",
                        scale=alt.Scale(
                            domain=list(SEGMENT_COLORS),
                            range=list(SEGMENT_COLORS.values()),
                        ),
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip("segmen:N", title="Segment"),
                        alt.Tooltip("jumlah:Q", title="Employees"),
                        alt.Tooltip("persen:Q", title="Share", format=".1f"),
                    ],
                )
                .properties(height=230)
            )
            st.altair_chart(segment_chart, use_container_width=True)

            st.markdown(
                '<div class="chart-footnote">Segment labels are ordered by average tenure.</div>',
                unsafe_allow_html=True,
            )

    with chart_right:
        with st.container(border=True):
            render_section_heading(
                "Tenure Distribution",
                "Sebaran lama bekerja di perusahaan saat ini.",
            )

            tenure_chart = (
                alt.Chart(employees)
                .mark_bar(color="#3478F6", cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X(
                        "YearsAtCompany:Q",
                        bin=alt.Bin(maxbins=18),
                        title="Years at Company",
                        axis=alt.Axis(grid=False),
                    ),
                    y=alt.Y("count():Q", title="Number of Employees"),
                    tooltip=[
                        alt.Tooltip("count():Q", title="Employees"),
                        alt.Tooltip("YearsAtCompany:Q", bin=alt.Bin(maxbins=18), title="Tenure range"),
                    ],
                )
                .properties(height=230)
            )
            st.altair_chart(tenure_chart, use_container_width=True)

    st.write("")
    lower_left, lower_right = st.columns([1.35, 1], gap="large")

    with lower_left:
        with st.container(border=True):
            render_section_heading(
                "Cluster Profile",
                "Rata-rata karakteristik masa kerja per segmen.",
            )
            profile_view = profiles[
                [
                    "segmen",
                    "jumlah",
                    "persen",
                    "YearsAtCompany",
                    "YearsInCurrentRole",
                    "TotalWorkingYears",
                ]
            ].rename(
                columns={
                    "segmen": "Segment",
                    "jumlah": "Employees",
                    "persen": "Share (%)",
                    "YearsAtCompany": "Company Tenure",
                    "YearsInCurrentRole": "Current Role",
                    "TotalWorkingYears": "Total Experience",
                }
            )
            st.dataframe(
                profile_view,
                hide_index=True,
                use_container_width=True,
                height=220,
            )

    with lower_right:
        with st.container(border=True):
            render_section_heading(
                "Department Overview",
                "Komposisi karyawan berdasarkan departemen.",
            )
            department_data = (
                employees.groupby("Department", as_index=False)
                .size()
                .rename(columns={"size": "Employees"})
                .sort_values("Employees", ascending=False)
            )
            department_chart = (
                alt.Chart(department_data)
                .mark_bar(color="#20A8D8", cornerRadiusEnd=5)
                .encode(
                    x=alt.X("Employees:Q", title="Employees", axis=alt.Axis(grid=False)),
                    y=alt.Y("Department:N", sort="-x", title=None),
                    tooltip=["Department", "Employees"],
                )
                .properties(height=220)
            )
            st.altair_chart(department_chart, use_container_width=True)

    if search_query:
        st.info(
            "Pencarian global tersedia pada halaman Employee Directory. "
            f"Kata kunci saat ini: **{search_query}**"
        )


# ============================================================
# EMPLOYEE DIRECTORY
# ============================================================

def render_employee_directory(
    employees: pd.DataFrame,
    global_search: str,
) -> None:
    render_page_heading(
        "PEOPLE / EMPLOYEE DIRECTORY",
        "Employee Directory",
        "Cari, filter, dan tinjau karakteristik karyawan berdasarkan hasil segmentasi.",
    )

    filter_col1, filter_col2, filter_col3 = st.columns([1.2, 1.2, 1.6])

    departments = ["All Departments"] + sorted(
        employees["Department"].dropna().astype(str).unique().tolist()
    )
    segments = ["All Segments"] + sorted(
        employees["segmen"].dropna().astype(str).unique().tolist()
    )

    with filter_col1:
        selected_department = st.selectbox("Department", departments)
    with filter_col2:
        selected_segment = st.selectbox("Employee Segment", segments)
    with filter_col3:
        search = st.text_input(
            "Search Employee",
            value=global_search,
            placeholder="Employee ID, role, department...",
        )

    filtered = employees.copy()

    if selected_department != "All Departments":
        filtered = filtered[filtered["Department"] == selected_department]

    if selected_segment != "All Segments":
        filtered = filtered[filtered["segmen"] == selected_segment]

    if search.strip():
        query = search.strip().lower()
        searchable_columns = [
            col
            for col in ["EmployeeNumber", "Department", "JobRole", "segmen"]
            if col in filtered.columns
        ]
        mask = pd.Series(False, index=filtered.index)
        for column in searchable_columns:
            mask |= filtered[column].astype(str).str.lower().str.contains(
                query, regex=False, na=False
            )
        filtered = filtered[mask]

    result_columns = [
        "EmployeeNumber",
        "Department",
        "JobRole",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "TotalWorkingYears",
        "cluster",
        "segmen",
    ]
    result_columns = [col for col in result_columns if col in filtered.columns]

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Filtered Employees", format_number(len(filtered)))
    kpi2.metric(
        "Average Company Tenure",
        f"{filtered['YearsAtCompany'].mean():.1f} years" if not filtered.empty else "—",
    )
    kpi3.metric(
        "Departments",
        str(filtered["Department"].nunique()) if not filtered.empty else "0",
    )

    with st.container(border=True):
        render_section_heading("Employee Records", "Data ditampilkan berdasarkan filter yang dipilih.")
        st.dataframe(
            filtered[result_columns].sort_values(
                "YearsAtCompany", ascending=False
            ),
            hide_index=True,
            use_container_width=True,
            height=470,
        )

        st.download_button(
            "Download filtered data",
            data=filtered[result_columns].to_csv(index=False).encode("utf-8-sig"),
            file_name="employee_directory.csv",
            mime="text/csv",
            use_container_width=False,
        )


# ============================================================
# CLUSTER ANALYTICS
# ============================================================

def render_cluster_analytics(
    employees: pd.DataFrame,
    profiles: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> None:
    render_page_heading(
        "MACHINE LEARNING / CLUSTER ANALYTICS",
        "Cluster Analytics",
        "Analisis pola cluster, karakteristik segmen, dan evaluasi model K-Means.",
    )

    with st.container(border=True):
        render_section_heading(
            "PCA Cluster Visualization",
            "Proyeksi dua dimensi dari fitur yang digunakan model. PC1 dan PC2 bukan satuan tahun.",
        )
        scatter = (
            alt.Chart(employees)
            .mark_circle(size=62, opacity=0.68, stroke="white", strokeWidth=0.4)
            .encode(
                x=alt.X("PC1:Q", title="Principal Component 1"),
                y=alt.Y("PC2:Q", title="Principal Component 2"),
                color=alt.Color(
                    "segmen:N",
                    scale=alt.Scale(
                        domain=list(SEGMENT_COLORS),
                        range=list(SEGMENT_COLORS.values()),
                    ),
                    title="Employee Segment",
                ),
                tooltip=[
                    "EmployeeNumber",
                    "segmen",
                    "YearsAtCompany",
                    "YearsInCurrentRole",
                    "TotalWorkingYears",
                ],
            )
            .interactive()
            .properties(height=420)
        )
        st.altair_chart(scatter, use_container_width=True)

    st.write("")
    profile_col, metrics_col = st.columns([1.25, 1], gap="large")

    with profile_col:
        with st.container(border=True):
            render_section_heading("Cluster Characteristics", "Rata-rata fitur pada setiap cluster.")
            profile_display = profiles.rename(
                columns={
                    "segmen": "Segment",
                    "jumlah": "Employees",
                    "persen": "Share (%)",
                    "YearsAtCompany": "Years at Company",
                    "YearsInCurrentRole": "Years in Current Role",
                    "TotalWorkingYears": "Total Working Years",
                }
            )
            st.dataframe(profile_display, hide_index=True, use_container_width=True)

    with metrics_col:
        with st.container(border=True):
            render_section_heading("Selected Model", "Model yang digunakan pada aplikasi.")
            st.markdown(
                """
                <div class="model-summary">
                    <div class="model-summary-label">ALGORITHM</div>
                    <div class="model-summary-value">K-Means Clustering</div>
                    <div class="model-summary-meta">StandardScaler · 3 features · 4 clusters</div>
                </div>
                <div class="feature-chip-list">
                    <span>YearsAtCompany</span>
                    <span>YearsInCurrentRole</span>
                    <span>TotalWorkingYears</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    with st.container(border=True):
        render_section_heading(
            "Model Evaluation",
            "Perbandingan nilai evaluasi untuk kandidat jumlah cluster K.",
        )

        best_silhouette = evaluation.loc[evaluation["silhouette"].idxmax()]
        best_db = evaluation.loc[evaluation["davies_bouldin"].idxmin()]

        eval1, eval2, eval3 = st.columns(3)
        eval1.metric(
            "Highest Silhouette",
            f"{best_silhouette['silhouette']:.3f}",
            f"K = {int(best_silhouette['k'])}",
        )
        eval2.metric(
            "Lowest Davies-Bouldin",
            f"{best_db['davies_bouldin']:.3f}",
            f"K = {int(best_db['k'])}",
        )
        eval3.metric("Selected K", "4", "Current trained model")

        chart1, chart2 = st.columns(2)
        with chart1:
            st.markdown("**Silhouette Score**")
            st.caption("Semakin tinggi umumnya semakin baik.")
            st.line_chart(evaluation.set_index("k")[["silhouette"]])
        with chart2:
            st.markdown("**Davies-Bouldin Index**")
            st.caption("Semakin rendah umumnya semakin baik.")
            st.line_chart(evaluation.set_index("k")[["davies_bouldin"]])

        st.markdown("**Elbow Method — Inertia**")
        st.caption("Penurunan inertia diamati untuk membantu memilih kandidat K.")
        st.line_chart(evaluation.set_index("k")[["inertia"]])

        st.dataframe(evaluation, hide_index=True, use_container_width=True)

    st.caption(
        "Interpretasi kualitas cluster perlu mempertimbangkan metrik evaluasi, "
        "distribusi data, dan konteks bisnis; tidak ditentukan oleh satu metrik saja."
    )


# ============================================================
# PREDICTION
# ============================================================

def predict_one(
    values: dict[str, float],
    assets: tuple[Any, Any, dict[str, Any]],
) -> dict[str, Any]:
    model, scaler, info = assets
    features = info["fitur"]

    missing = [feature for feature in features if feature not in values]
    if missing:
        raise ValueError("Fitur prediksi tidak lengkap: " + ", ".join(missing))

    row = pd.DataFrame([[float(values[feature]) for feature in features]], columns=features)

    if not np.isfinite(row.to_numpy()).all():
        raise ValueError("Seluruh input harus berupa angka yang valid.")
    if (row < 0).any().any():
        raise ValueError("Nilai masa kerja tidak boleh negatif.")

    if values["YearsInCurrentRole"] > values["YearsAtCompany"]:
        raise ValueError("YearsInCurrentRole tidak boleh melebihi YearsAtCompany.")
    if values["YearsAtCompany"] > values["TotalWorkingYears"]:
        raise ValueError("YearsAtCompany tidak boleh melebihi TotalWorkingYears.")

    raw_cluster = int(model.predict(scaler.transform(row))[0])
    cluster_order = int(info["peta"][raw_cluster])
    segment = info["nama_segmen"][cluster_order]

    return {"cluster": cluster_order, "segmen": segment}


def render_prediction() -> None:
    render_page_heading(
        "EMPLOYEE SEGMENTATION / PREDICTION",
        "Predict Employee Segment",
        "Prediksi segmen karyawan baru menggunakan model K-Means yang telah dilatih.",
    )

    try:
        assets = load_model_assets()
    except Exception as exc:
        st.error(f"Model tidak dapat dimuat: {exc}")
        st.stop()

    model, scaler, info = assets
    features = info["fitur"]

    st.info(
        "Fitur model: " + ", ".join(features) + ". "
        "Masukkan nilai dalam satuan tahun."
    )

    manual_tab, batch_tab = st.tabs(["Manual Prediction", "Batch Prediction"])

    with manual_tab:
        with st.container(border=True):
            render_section_heading("Employee Profile", "Lengkapi tiga indikator masa kerja berikut.")
            with st.form("employee_prediction_form"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    company_years = st.number_input(
                        "Years at Company", min_value=0, max_value=60, value=5, step=1
                    )
                with col2:
                    role_years = st.number_input(
                        "Years in Current Role", min_value=0, max_value=60, value=3, step=1
                    )
                with col3:
                    total_years = st.number_input(
                        "Total Working Years", min_value=0, max_value=60, value=10, step=1
                    )

                submitted = st.form_submit_button(
                    "Run Prediction", type="primary", use_container_width=True
                )

            if submitted:
                values = {
                    "YearsAtCompany": company_years,
                    "YearsInCurrentRole": role_years,
                    "TotalWorkingYears": total_years,
                }
                try:
                    result = predict_one(values, assets)
                    st.success("Prediksi berhasil dijalankan.")
                    result_col1, result_col2 = st.columns(2)
                    result_col1.metric("Cluster ID", result["cluster"])
                    result_col2.metric("Employee Segment", result["segmen"])
                    st.markdown(
                        '<div class="prediction-note">Hasil ini merupakan output pengelompokan model, bukan keputusan otomatis terkait status atau kinerja karyawan.</div>',
                        unsafe_allow_html=True,
                    )
                except ValueError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Prediksi gagal: {exc}")

    with batch_tab:
        with st.container(border=True):
            render_section_heading("Batch Prediction", "Unggah CSV dengan kolom fitur yang digunakan model.")
            st.markdown(
                "Kolom wajib: `YearsAtCompany`, `YearsInCurrentRole`, `TotalWorkingYears`."
            )
            uploaded = st.file_uploader("Upload employee CSV", type=["csv"], key="batch_upload")

            if uploaded is not None:
                try:
                    batch = pd.read_csv(uploaded)
                    missing = [feature for feature in features if feature not in batch.columns]

                    if missing:
                        st.error("Kolom berikut belum tersedia: " + ", ".join(missing))
                    else:
                        for feature in features:
                            batch[feature] = pd.to_numeric(batch[feature], errors="coerce")

                        invalid = batch[features].isna().any(axis=1)
                        invalid |= (batch[features] < 0).any(axis=1)
                        invalid |= batch["YearsInCurrentRole"] > batch["YearsAtCompany"]
                        invalid |= batch["YearsAtCompany"] > batch["TotalWorkingYears"]

                        if invalid.any():
                            st.error(
                                f"Terdapat {int(invalid.sum())} baris dengan nilai kosong, "
                                "negatif, atau hubungan masa kerja yang tidak valid. "
                                "Perbaiki data lalu unggah kembali."
                            )
                        else:
                            scaled = scaler.transform(batch[features])
                            raw_labels = model.predict(scaled)
                            ordered_labels = [int(info["peta"][int(label)]) for label in raw_labels]
                            batch["cluster"] = ordered_labels
                            batch["segmen"] = [
                                info["nama_segmen"][label] for label in ordered_labels
                            ]

                            st.success(f"Berhasil memproses {len(batch):,} baris data.")
                            st.dataframe(batch, hide_index=True, use_container_width=True)
                            st.download_button(
                                "Download prediction results",
                                data=batch.to_csv(index=False).encode("utf-8-sig"),
                                file_name="employee_clustering_predictions.csv",
                                mime="text/csv",
                            )
                except Exception as exc:
                    st.error(f"File CSV tidak dapat diproses: {exc}")


# ============================================================
# MODEL INFORMATION
# ============================================================

def render_model_information() -> None:
    render_page_heading(
        "DOCUMENTATION / MODEL DETAILS",
        "Model Information",
        "Informasi algoritma, fitur, proses preprocessing, dan batasan interpretasi model.",
    )

    try:
        _, _, info = load_model_assets()
    except Exception as exc:
        st.error(f"Informasi model tidak dapat dimuat: {exc}")
        st.stop()

    with st.container(border=True):
        render_section_heading("Model Configuration")
        col1, col2, col3 = st.columns(3)
        col1.metric("Algorithm", "K-Means")
        col2.metric("Number of Clusters", str(len(info.get("nama_segmen", {}))))
        col3.metric("Input Features", str(len(info.get("fitur", []))))

        st.markdown("**Features used by the model**")
        for feature in info.get("fitur", []):
            st.markdown(f"- `{feature}`")

    with st.container(border=True):
        render_section_heading("Methodology")
        st.markdown(
            """
            1. **Data preparation** — dataset dibersihkan sebelum proses training.
            2. **Feature selection** — model menggunakan `YearsAtCompany`, `YearsInCurrentRole`, dan `TotalWorkingYears`.
            3. **Scaling** — `StandardScaler` menyetarakan skala fitur.
            4. **Clustering** — K-Means membentuk empat cluster.
            5. **Cluster ordering** — label cluster diurutkan berdasarkan rata-rata `YearsAtCompany` untuk membantu interpretasi senioritas.
            6. **Evaluation** — kandidat jumlah cluster dibandingkan menggunakan inertia, Silhouette Score, dan Davies-Bouldin Index.
            7. **PCA** — digunakan untuk visualisasi dua dimensi, bukan sebagai input utama model prediksi.
            """
        )

    st.warning(
        "Model ini melakukan segmentasi berdasarkan tiga indikator masa kerja. "
        "Label Junior, Menengah, dan Senior merupakan nama interpretatif berdasarkan urutan cluster, "
        "bukan batas kategori HR yang berlaku universal."
    )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

def main() -> None:
    try:
        employees = load_employee_data()
        profiles = load_cluster_profiles()
        evaluation = load_evaluation()
    except Exception as exc:
        st.error(f"Aplikasi gagal memuat data: {exc}")
        st.info(
            "Pastikan hasil_cluster.csv, profil_cluster.csv, evaluasi_k.csv, "
            "dan file model berada di direktori utama project."
        )
        st.stop()

    page = render_sidebar()
    global_search = render_topbar()

    if page == "Dashboard":
        render_dashboard(employees, profiles, global_search)
    elif page == "Employee Directory":
        render_employee_directory(employees, global_search)
    elif page == "Cluster Analytics":
        render_cluster_analytics(employees, profiles, evaluation)
    elif page == "Predict Employee":
        render_prediction()
    elif page == "Model Information":
        render_model_information()

    st.markdown(
        """
        <div class="app-footer">
            <span>EMPLOYEE TENURE INSIGHTS</span>
            <span>Python · Streamlit · K-Means Clustering</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
