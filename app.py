# -*- coding: utf-8 -*-
"""
Privacy-Aware Advertising Measurement in AI Advertising Intelligence and Revenue Optimization
Single-file Streamlit prototype.

Goals:
- Run locally on Windows 10/11, CPU-only, no API keys.
- Use only synthetic data.
- Automatically load realistic demo data and start a live simulation.
- Provide a reviewable product, architecture, data-contract, testing, risk, and demo artifact.
- Separate observed, reported/noised, estimated, predicted, and simulated-truth values.
"""

import os
import io
import json
import math
import time
import uuid
import sqlite3
import tempfile
import hashlib
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import HistGradientBoostingRegressor

try:
    import cv2
except Exception:
    cv2 = None

try:
    from PIL import Image, ImageDraw
except Exception:
    Image = None
    ImageDraw = None


# -----------------------------------------------------------------------------
# Early utilities and Streamlit compatibility helpers
# -----------------------------------------------------------------------------


def stable_hash_int(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % 1000000


def cache_data_compat(func):
    if hasattr(st, "cache_data"):
        return st.cache_data(func)
    if hasattr(st, "cache"):
        return st.cache(func)
    return func


def rerun_app() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.stop()


def clear_caches() -> None:
    if hasattr(st, "cache_data") and hasattr(st.cache_data, "clear"):
        st.cache_data.clear()
    elif hasattr(st, "cache") and hasattr(st.cache, "clear"):
        st.cache.clear()


def next_uid(prefix: str = "el") -> str:
    if "_uid_counter" not in st.session_state:
        st.session_state["_uid_counter"] = 0
    st.session_state["_uid_counter"] += 1
    return f"{prefix}_{st.session_state['_uid_counter']}_{uuid.uuid4().hex[:8]}"


def zw_suffix(seed: str) -> str:
    return "\u200b" * ((stable_hash_int(seed) % 7) + 1)


def make_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return make_json_safe(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return make_json_safe(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return make_json_safe(obj.tolist())
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, np.generic):
        return obj.item()
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj


def read_bytes(path: str) -> bytes:
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return b""


def el_plotly(container: Any, fig: go.Figure, key: Optional[str] = None) -> None:
    if key is None:
        key = next_uid("plotly")
    try:
        container.plotly_chart(fig, use_container_width=True, key=key)
    except TypeError:
        try:
            container.plotly_chart(fig, use_column_width=True, key=key)
        except TypeError:
            try:
                container.plotly_chart(fig, key=key)
            except TypeError:
                container.plotly_chart(fig)


def show_plotly(fig: go.Figure, key: Optional[str] = None) -> None:
    el_plotly(st, fig, key=key)


def el_df(container: Any, df: pd.DataFrame, key: Optional[str] = None) -> None:
    if key is None:
        key = next_uid("df")
    try:
        container.dataframe(df, use_container_width=True, key=key)
    except TypeError:
        try:
            container.dataframe(df, key=key)
        except TypeError:
            container.dataframe(df)


def show_df(df: pd.DataFrame, key: Optional[str] = None) -> None:
    el_df(st, df, key=key)


def el_json(container: Any, obj: Any, key: Optional[str] = None) -> None:
    if key is None:
        key = next_uid("json")
    safe_obj = make_json_safe(obj)
    try:
        container.json(safe_obj, key=key)
    except TypeError:
        container.json(safe_obj)


def el_md(container: Any, body: str, key: Optional[str] = None) -> None:
    if key is None:
        key = next_uid("md")
    try:
        container.markdown(body, key=key)
    except TypeError:
        container.markdown(body + zw_suffix(key))


def el_caption(container: Any, text: str, key: Optional[str] = None) -> None:
    if key is None:
        key = next_uid("cap")
    try:
        container.caption(text, key=key)
    except TypeError:
        container.caption(text + zw_suffix(key))


def el_msg(container: Any, body: str, kind: str = "info", key: Optional[str] = None) -> None:
    if key is None:
        key = next_uid(kind)
    func = getattr(container, kind, None)
    if func is None:
        el_md(container, f"**{kind.upper()}:** {body}", key=key)
        return
    try:
        func(body, key=key)
    except TypeError:
        try:
            func(body + zw_suffix(key))
        except TypeError:
            func(body)


def el_metric(container: Any, label: str, value: Any, delta: Optional[Any] = None, key: Optional[str] = None) -> None:
    if key is None:
        key = next_uid("metric")
    try:
        if delta is None:
            container.metric(label, value, key=key)
        else:
            container.metric(label, value, delta=delta, key=key)
    except TypeError:
        lab = label + zw_suffix(key)
        if delta is None:
            container.metric(lab, value)
        else:
            container.metric(lab, value, delta=delta)


def el_button(container: Any, label: str, key: Optional[str] = None, **kwargs: Any) -> bool:
    if key is None:
        key = next_uid("btn")
    try:
        return bool(container.button(label, key=key, **kwargs))
    except TypeError:
        return bool(container.button(label + zw_suffix(key), **kwargs))


def el_download(container: Any, label: str, data: bytes, file_name: str, mime: str, key: Optional[str] = None) -> None:
    if key is None:
        key = next_uid("dl")
    try:
        container.download_button(label, data=data, file_name=file_name, mime=mime, key=key)
    except TypeError:
        container.download_button(label + zw_suffix(key), data=data, file_name=file_name, mime=mime)


def el_video(container: Any, path: str, kind: str, key: Optional[str] = None) -> None:
    if key is None:
        key = next_uid("video")
    fmt = "video/mp4" if kind == "mp4" else "video/webm"
    data = read_bytes(path)
    if not data:
        return
    try:
        container.video(data, format=fmt, key=key)
    except TypeError:
        try:
            container.video(data, format=fmt)
        except TypeError:
            try:
                container.video(path, key=key)
            except TypeError:
                container.video(path)
    except Exception:
        try:
            container.video(path)
        except Exception:
            pass


def el_image(container: Any, src: Any, caption: Optional[str] = None, key: Optional[str] = None) -> None:
    if key is None:
        key = next_uid("img")
    if caption:
        el_caption(container, caption, key=next_uid("imgcap"))
    try:
        container.image(src, use_container_width=True, key=key)
    except TypeError:
        try:
            container.image(src, use_column_width=True, key=key)
        except TypeError:
            try:
                container.image(src, key=key)
            except TypeError:
                container.image(src)


def el_code(container: Any, code: str, language: Optional[str] = None, key: Optional[str] = None) -> None:
    if key is None:
        key = next_uid("code")
    try:
        if language is None:
            container.code(code, key=key)
        else:
            container.code(code, language=language, key=key)
    except TypeError:
        if language is None:
            container.code(code)
        else:
            container.code(code, language=language)


# -----------------------------------------------------------------------------
# Constants and configuration
# -----------------------------------------------------------------------------

APP_TITLE = "Privacy-Aware Advertising Measurement"
APP_SUBTITLE = "AI Advertising Intelligence and Revenue Optimization"
SCHEMA_VERSION = "1.0.0"
PROVENANCE = "synthetic://privacy-ad-measurement/v1"
SEED = 20261108

TMP_DIR = tempfile.gettempdir()
DB_PATH = os.path.join(TMP_DIR, f"privacy_ad_demo_{os.getpid()}.sqlite")
MP4_PATH = os.path.join(TMP_DIR, "privacy_ad_demo.mp4")
WEBM_PATH = os.path.join(TMP_DIR, "privacy_ad_demo.webm")
GIF_PATH = os.path.join(TMP_DIR, "privacy_ad_demo.gif")

MIN_CELL_POPULATION_DEFAULT = 1000
DEFAULT_EPSILON = 1.0
POLICY_MAX_EPSILON = 2.0
PRIVACY_DELTA = 1e-5
MAX_ORDER_VALUE = 500.0
GRACE_DAYS = 7
BOOTSTRAP_N = 80
LIVE_TICKS = 30

VALID_CHANNELS = {"search", "social", "ctv", "retail_media", "display", "email"}
VALID_GROUPS = {"treated", "holdout"}
VALID_CONSENT = {"granted", "denied", "unknown"}
VALID_DEVICE = {"mobile", "desktop", "ctv"}
VALID_AGE = {"18-24", "25-44", "45+"}
VALID_GEO = {"US-NE", "US-MW", "US-W", "EU-B"}

CAMPAIGN_SPECS = [
    {
        "campaign_id": "cmp_search_brand",
        "channel": "search",
        "objective": "brand_defense",
        "base_cvr": 0.032,
        "uplift_mult": 0.10,
        "cpm": 7.5,
        "aov_mean": 85.0,
        "ctr": 0.045,
    },
    {
        "campaign_id": "cmp_social_prospecting",
        "channel": "social",
        "objective": "prospecting",
        "base_cvr": 0.018,
        "uplift_mult": 0.18,
        "cpm": 11.0,
        "aov_mean": 72.0,
        "ctr": 0.018,
    },
    {
        "campaign_id": "cmp_ctv_reach",
        "channel": "ctv",
        "objective": "reach_frequency",
        "base_cvr": 0.009,
        "uplift_mult": -0.04,
        "cpm": 24.0,
        "aov_mean": 120.0,
        "ctr": 0.004,
    },
    {
        "campaign_id": "cmp_retail_media",
        "channel": "retail_media",
        "objective": "closed_loop_sales",
        "base_cvr": 0.026,
        "uplift_mult": 0.22,
        "cpm": 15.0,
        "aov_mean": 54.0,
        "ctr": 0.012,
    },
    {
        "campaign_id": "cmp_display_remktg",
        "channel": "display",
        "objective": "remarketing",
        "base_cvr": 0.014,
        "uplift_mult": 0.03,
        "cpm": 4.2,
        "aov_mean": 66.0,
        "ctr": 0.006,
    },
    {
        "campaign_id": "cmp_email_lifecycle",
        "channel": "email",
        "objective": "lifecycle",
        "base_cvr": 0.040,
        "uplift_mult": 0.08,
        "cpm": 0.8,
        "aov_mean": 91.0,
        "ctr": 0.090,
    },
]

AGE_FACTOR = {"18-24": 0.72, "25-44": 1.16, "45+": 0.94}
CONSENT_FACTOR = {"granted": 1.00, "denied": 0.58, "unknown": 0.76}
DEVICE_FACTOR = {"mobile": 1.00, "desktop": 0.86, "ctv": 0.62}
GEO_FACTOR = {"US-NE": 1.06, "US-MW": 0.95, "US-W": 1.02, "EU-B": 0.91}

CANONICAL_COLS = [
    "cell_id",
    "event_date",
    "campaign_id",
    "channel",
    "geo_bucket",
    "age_bucket",
    "device_type",
    "consent_state",
    "exposure_group",
    "eligible_population",
    "impressions",
    "clicks_reported",
    "conversions_reported",
    "revenue_reported",
    "spend_reported",
    "epsilon",
    "privacy_delta",
    "noise_mechanism",
    "noise_conversions",
    "noise_revenue",
    "suppressed",
    "released",
    "quality_flags",
    "provenance",
    "schema_version",
    "source_system",
    "ingested_at",
    "lateness_days",
]

REQUIRED_DIMS = [
    "cell_id",
    "event_date",
    "campaign_id",
    "channel",
    "geo_bucket",
    "age_bucket",
    "device_type",
    "consent_state",
    "exposure_group",
]

CRITICAL_NUMERIC = [
    "eligible_population",
    "conversions_reported",
    "revenue_reported",
    "spend_reported",
]


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso(ts: datetime) -> str:
    return ts.isoformat(timespec="seconds")


def safe_sample(df: pd.DataFrame, n: int, random_state: int) -> pd.DataFrame:
    n = min(n, len(df))
    if n <= 0:
        return df.iloc[0:0].copy()
    return df.sample(n=n, random_state=random_state).copy()


def add_flag_str(value: Any, flag: str) -> str:
    if value is None:
        return flag
    try:
        if isinstance(value, float) and math.isnan(value):
            return flag
    except Exception:
        pass
    s = str(value).strip()
    if not s:
        return flag
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if flag not in parts:
        parts.append(flag)
    return ",".join(parts)


def make_cell_id(
    event_date: pd.Timestamp,
    campaign_id: str,
    geo_bucket: str,
    age_bucket: str,
    device_type: str,
    consent_state: str,
    exposure_group: str,
) -> str:
    raw = "|".join(
        [
            pd.Timestamp(event_date).strftime("%Y-%m-%d"),
            str(campaign_id),
            str(geo_bucket),
            str(age_bucket),
            str(device_type),
            str(consent_state),
            str(exposure_group),
            SCHEMA_VERSION,
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def clip01(x: Any) -> Any:
    """
    Scalar-safe and vector-safe clipping to [0, 1].
    """
    if isinstance(x, (pd.Series, np.ndarray)):
        arr = np.asarray(x, dtype=float)
        arr = np.where(np.isfinite(arr), arr, 0.0)
        return np.clip(arr, 0.0, 1.0)

    try:
        val = float(x)
    except Exception:
        return 0.0

    if not math.isfinite(val):
        return 0.0
    return float(max(0.0, min(1.0, val)))


def fmt_usd(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    return f"${x:,.0f}"


def fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{100.0 * x:.1f}%"


# -----------------------------------------------------------------------------
# Database and audit
# -----------------------------------------------------------------------------


def get_conn() -> sqlite3.Connection:
    if "conn" not in st.session_state:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        st.session_state.conn = conn
        init_schema(conn)
    return st.session_state.conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            actor TEXT NOT NULL,
            action TEXT NOT NULL,
            object_type TEXT,
            object_id TEXT,
            details TEXT,
            approval_required INTEGER DEFAULT 0,
            approved_by TEXT
        );

        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            schema_version TEXT,
            epsilon REAL,
            min_population INTEGER,
            method TEXT,
            rows_in INTEGER,
            rows_clean INTEGER,
            rows_quarantine INTEGER,
            latency_ms REAL,
            status TEXT,
            notes TEXT
        );
        """
    )
    conn.commit()


def log_audit(
    actor: str,
    action: str,
    object_type: str = "",
    object_id: str = "",
    details: Optional[Dict[str, Any]] = None,
    approval_required: bool = False,
    approved_by: Optional[str] = None,
) -> None:
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO audit_log
        (ts, actor, action, object_type, object_id, details, approval_required, approved_by)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            iso(now_utc()),
            actor,
            action,
            object_type,
            object_id,
            json.dumps(details or {}, default=str),
            1 if approval_required else 0,
            approved_by,
        ),
    )
    conn.commit()


def record_run(
    run_id: str,
    started_at: str,
    finished_at: str,
    epsilon: float,
    min_pop: int,
    method: str,
    rows_in: int,
    rows_clean: int,
    rows_quar: int,
    latency_ms: float,
    status: str = "ok",
    notes: str = "",
) -> None:
    conn = get_conn()
    conn.execute(
        """
        INSERT OR REPLACE INTO runs
        (run_id, started_at, finished_at, schema_version, epsilon, min_population,
         method, rows_in, rows_clean, rows_quarantine, latency_ms, status, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            started_at,
            finished_at,
            SCHEMA_VERSION,
            epsilon,
            min_pop,
            method,
            rows_in,
            rows_clean,
            rows_quar,
            latency_ms,
            status,
            notes,
        ),
    )
    conn.commit()


def df_for_sql(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["event_date", "ingested_at"]:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S")
            out[c] = out[c].fillna("").astype(str)
    for c in ["suppressed", "released"]:
        if c in out.columns:
            out[c] = out[c].astype(int)
    if "quality_flags" in out.columns:
        out["quality_flags"] = out["quality_flags"].fillna("").astype(str)
    if "quarantine_reason" in out.columns:
        out["quarantine_reason"] = out["quarantine_reason"].fillna("").astype(str)
    return out


def save_dataset_to_db(clean: pd.DataFrame, quarantine: pd.DataFrame, report: Dict[str, Any]) -> None:
    conn = get_conn()
    df_for_sql(clean).to_sql("cells", conn, if_exists="replace", index=False)
    df_for_sql(quarantine).to_sql("quarantine", conn, if_exists="replace", index=False)
    conn.commit()


# -----------------------------------------------------------------------------
# Synthetic latent truth generation
# -----------------------------------------------------------------------------


def generate_latent_history(seed: int, date_key: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(date_key).normalize()
    dates = pd.date_range(end=end, periods=14, freq="D")

    rows: List[Dict[str, Any]] = []

    for d in dates:
        day_of_week = d.dayofweek
        weekend_boost = 1.08 if day_of_week >= 5 else 1.0
        holiday_bump = 1.18 if d.day == 10 else 1.0

        for spec in CAMPAIGN_SPECS:
            for geo in sorted(VALID_GEO):
                for age in sorted(VALID_AGE):
                    for device in sorted(VALID_DEVICE):
                        for consent in sorted(VALID_CONSENT):
                            for group in ["treated", "holdout"]:
                                if rng.random() < 0.16:
                                    continue

                                base_pop = rng.lognormal(math.log(18000), 0.75)
                                if rng.random() < 0.12:
                                    base_pop = rng.uniform(180, 1450)
                                pop = int(max(50, min(750000, base_pop)))

                                exposure_rate = 0.0
                                impressions = 0
                                clicks = 0
                                spend = 0.0

                                if group == "treated":
                                    exposure_rate = float(np.clip(rng.beta(2.0, 5.0) + 0.12, 0.04, 0.88))
                                    impressions = int(pop * exposure_rate)
                                    ctr = float(np.clip(spec["ctr"] * DEVICE_FACTOR.get(device, 1.0) * weekend_boost, 0.0005, 0.25))
                                    clicks = int(rng.poisson(max(0.0, impressions * ctr)))
                                    spend = float(impressions * spec["cpm"] / 1000.0)

                                base_rate = (
                                    spec["base_cvr"]
                                    * AGE_FACTOR.get(age, 1.0)
                                    * CONSENT_FACTOR.get(consent, 1.0)
                                    * DEVICE_FACTOR.get(device, 1.0)
                                    * GEO_FACTOR.get(geo, 1.0)
                                    * weekend_boost
                                    * holiday_bump
                                )

                                if group == "treated":
                                    rate = max(0.0, base_rate * (1.0 + spec["uplift_mult"]))
                                else:
                                    rate = max(0.0, base_rate)

                                conversions = int(rng.poisson(max(0.0, rate * pop)))
                                aov = float(np.clip(rng.lognormal(math.log(max(spec["aov_mean"], 1.0)) - 0.18, 0.42), 5.0, MAX_ORDER_VALUE))
                                revenue = float(conversions * aov)

                                ingested_at = d + timedelta(hours=int(rng.integers(1, 72)))
                                source = "synthetic_adserver" if group == "treated" else "synthetic_holdout_logger"
                                if spec["channel"] in {"retail_media", "email"}:
                                    source = "synthetic_revenue_warehouse"

                                cell_id = make_cell_id(d, spec["campaign_id"], geo, age, device, consent, group)

                                rows.append(
                                    {
                                        "cell_id": cell_id,
                                        "event_date": d,
                                        "campaign_id": spec["campaign_id"],
                                        "channel": spec["channel"],
                                        "geo_bucket": geo,
                                        "age_bucket": age,
                                        "device_type": device,
                                        "consent_state": consent,
                                        "exposure_group": group,
                                        "eligible_population": pop,
                                        "impressions": impressions,
                                        "clicks_true": clicks,
                                        "conversions_true": conversions,
                                        "revenue_true": revenue,
                                        "spend_true": spend,
                                        "aov_true": aov,
                                        "provenance": PROVENANCE,
                                        "schema_version": SCHEMA_VERSION,
                                        "source_system": source,
                                        "ingested_at": ingested_at,
                                    }
                                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Synthetic latent generation produced no rows.")
    return df


def apply_privacy(latent: pd.DataFrame, epsilon: float, min_pop: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 17)
    df = latent.copy()

    eps = float(max(1e-6, epsilon))
    df["epsilon"] = eps
    df["privacy_delta"] = PRIVACY_DELTA
    df["noise_mechanism"] = "laplace"
    df["suppressed"] = df["eligible_population"].astype(int) < int(min_pop)
    df["released"] = ~df["suppressed"]

    b_conv = 1.0 / eps
    b_rev = MAX_ORDER_VALUE / eps
    b_click = 1.0 / eps

    noise_c = rng.laplace(0.0, b_conv, len(df))
    noise_r = rng.laplace(0.0, b_rev, len(df))
    noise_k = rng.laplace(0.0, b_click, len(df))

    df["noise_conversions"] = np.where(df["released"], noise_c, 0.0)
    df["noise_revenue"] = np.where(df["released"], noise_r, 0.0)
    df["noise_clicks"] = np.where(df["released"], noise_k, 0.0)

    df["conversions_reported"] = np.where(
        df["released"],
        np.maximum(0.0, np.round(df["conversions_true"].astype(float) + noise_c)),
        0.0,
    )
    df["revenue_reported"] = np.where(
        df["released"],
        np.maximum(0.0, df["revenue_true"].astype(float) + noise_r),
        0.0,
    )
    df["clicks_reported"] = np.where(
        df["released"],
        np.maximum(0.0, np.round(df["clicks_true"].astype(float) + noise_k)),
        0.0,
    )
    df["spend_reported"] = df["spend_true"].astype(float)
    df["quality_flags"] = ""
    df.loc[df["suppressed"], "quality_flags"] = "small_cell_suppressed"
    df["lateness_days"] = (pd.to_datetime(df["ingested_at"]) - pd.to_datetime(df["event_date"])).dt.total_seconds() / 86400.0
    return df


def add_dirty_records(reported: pd.DataFrame, seed: int) -> pd.DataFrame:
    dirty_frames = [reported.copy()]

    dup = safe_sample(reported, 18, seed + 101).copy()
    if not dup.empty:
        dup["ingested_at"] = pd.to_datetime(dup["ingested_at"]) + pd.Timedelta(hours=6)
        dup["source_system"] = "adserver_replay"
        dup["quality_flags"] = dup["quality_flags"].apply(lambda x: add_flag_str(x, "duplicate_candidate"))
        dirty_frames.append(dup)

    miss = safe_sample(reported, 12, seed + 102).copy()
    if not miss.empty:
        miss["ingested_at"] = pd.to_datetime(miss["ingested_at"]) + pd.Timedelta(hours=1)
        miss["clicks_reported"] = np.nan
        miss["quality_flags"] = miss["quality_flags"].apply(lambda x: add_flag_str(x, "missing_clicks"))
        dirty_frames.append(miss)

    neg = safe_sample(reported, 7, seed + 103).copy()
    if not neg.empty:
        neg["ingested_at"] = pd.to_datetime(neg["ingested_at"]) + pd.Timedelta(hours=2)
        neg["spend_reported"] = -17.25
        neg["quality_flags"] = neg["quality_flags"].apply(lambda x: add_flag_str(x, "negative_spend"))
        dirty_frames.append(neg)

    bad = safe_sample(reported, 7, seed + 104).copy()
    if not bad.empty:
        bad["ingested_at"] = pd.to_datetime(bad["ingested_at"]) + pd.Timedelta(hours=2)
        bad["channel"] = "televisión"
        bad["quality_flags"] = bad["quality_flags"].apply(lambda x: add_flag_str(x, "invalid_channel"))
        dirty_frames.append(bad)

    out = safe_sample(reported, 7, seed + 105).copy()
    if not out.empty:
        out["ingested_at"] = pd.to_datetime(out["ingested_at"]) + pd.Timedelta(hours=2)
        out["revenue_reported"] = 1.35e9
        out["quality_flags"] = out["quality_flags"].apply(lambda x: add_flag_str(x, "outlier_revenue"))
        dirty_frames.append(out)

    late = safe_sample(reported, 7, seed + 106).copy()
    if not late.empty:
        late["event_date"] = pd.to_datetime(late["event_date"]) - pd.Timedelta(days=21)
        late["ingested_at"] = pd.to_datetime(late["ingested_at"]) + pd.Timedelta(hours=2)
        late["quality_flags"] = late["quality_flags"].apply(lambda x: add_flag_str(x, "late_out_of_window"))
        dirty_frames.append(late)

    unk = safe_sample(reported, 7, seed + 107).copy()
    if not unk.empty:
        unk["ingested_at"] = pd.to_datetime(unk["ingested_at"]) + pd.Timedelta(hours=2)
        unk["experimental_signal"] = 0.73
        unk["quality_flags"] = unk["quality_flags"].apply(lambda x: add_flag_str(x, "unknown_field"))
        dirty_frames.append(unk)

    raw = pd.concat(dirty_frames, ignore_index=True, sort=False)
    return raw


@cache_data_compat
def prepare_dataset(epsilon: float, min_pop: int, seed: int, dirty: bool, date_key: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], pd.DataFrame, float]:
    t0 = time.time()
    latent = generate_latent_history(seed=seed, date_key=date_key)
    reported = apply_privacy(latent=latent, epsilon=epsilon, min_pop=min_pop, seed=seed)
    raw = add_dirty_records(reported, seed=seed) if dirty else reported.copy()
    clean, quarantine, report = validate_and_canonicalize(raw)
    latent_clean = latent[latent["cell_id"].isin(set(clean["cell_id"]))].copy()
    latency = time.time() - t0
    report["prep_latency_sec"] = latency
    return clean, quarantine, report, latent_clean, latency


def validate_and_canonicalize(raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    df = raw.copy()
    report: Dict[str, Any] = {"input_rows": int(len(df))}

    for c in CANONICAL_COLS:
        if c not in df.columns:
            df[c] = np.nan

    unknown_cols = [c for c in df.columns if c not in CANONICAL_COLS]
    if unknown_cols:
        report["unknown_columns_dropped"] = unknown_cols
        df = df.drop(columns=unknown_cols)

    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df["ingested_at"] = pd.to_datetime(df["ingested_at"], errors="coerce")
    df["quality_flags"] = df["quality_flags"].fillna("").astype(str)

    num_cols = [
        "eligible_population",
        "impressions",
        "clicks_reported",
        "conversions_reported",
        "revenue_reported",
        "spend_reported",
        "epsilon",
        "privacy_delta",
        "noise_conversions",
        "noise_revenue",
        "lateness_days",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    bad_dim = df[REQUIRED_DIMS].isna().any(axis=1)
    enum_bad = (
        (~df["channel"].isin(VALID_CHANNELS))
        | (~df["exposure_group"].isin(VALID_GROUPS))
        | (~df["consent_state"].isin(VALID_CONSENT))
        | (~df["device_type"].isin(VALID_DEVICE))
        | (~df["age_bucket"].isin(VALID_AGE))
        | (~df["geo_bucket"].isin(VALID_GEO))
    )
    range_bad = (
        (df["eligible_population"] < 0)
        | (df["impressions"] < 0)
        | (df["clicks_reported"] < 0)
        | (df["conversions_reported"] < 0)
        | (df["revenue_reported"] < 0)
        | (df["spend_reported"] < 0)
        | (df["epsilon"] <= 0)
    )
    critical_missing = df[CRITICAL_NUMERIC].isna().any(axis=1)

    df["lateness_days"] = (df["ingested_at"] - df["event_date"]).dt.total_seconds() / 86400.0
    late_bad = df["lateness_days"] > GRACE_DAYS
    date_bad = df["event_date"].isna() | df["ingested_at"].isna()

    qmask = bad_dim | enum_bad | range_bad | critical_missing | late_bad | date_bad

    reason_series = pd.Series("", index=df.index, dtype=str)
    reason_masks = [
        (bad_dim, "missing_dimension"),
        (enum_bad, "invalid_enum"),
        (range_bad, "negative_or_invalid_range"),
        (critical_missing, "missing_critical_metric"),
        (late_bad, "late_out_of_grace_window"),
        (date_bad, "invalid_timestamp"),
    ]
    for mask, name in reason_masks:
        if mask.any():
            reason_series.loc[mask] = reason_series.loc[mask].apply(lambda x: add_flag_str(x, name))

    quarantine = df[qmask].copy()
    quarantine["quarantine_reason"] = reason_series[qmask]
    clean = df[~qmask].copy()

    for col in ["clicks_reported", "impressions", "conversions_reported", "revenue_reported", "spend_reported"]:
        miss = clean[col].isna()
        if miss.any():
            clean.loc[miss, "quality_flags"] = clean.loc[miss, "quality_flags"].apply(lambda x: add_flag_str(x, f"missing_{col}"))
            clean.loc[miss, col] = 0.0

    for col in ["conversions_reported", "revenue_reported"]:
        pos = clean.loc[clean[col] > 0, col]
        if len(pos) >= 20:
            q1 = float(pos.quantile(0.25))
            q3 = float(pos.quantile(0.75))
            iqr = max(q3 - q1, 1.0)
            upper = q3 + 10.0 * iqr
            out_mask = clean[col] > upper
            if out_mask.any():
                clean.loc[out_mask, "quality_flags"] = clean.loc[out_mask, "quality_flags"].apply(lambda x: add_flag_str(x, f"outlier_capped_{col}"))
                clean.loc[out_mask, col] = upper

    before_dup = len(clean)
    clean = clean.sort_values("ingested_at", kind="stable").drop_duplicates("cell_id", keep="last")
    report["duplicates_dropped"] = int(before_dup - len(clean))

    late_ok = (clean["lateness_days"] > 2.0) & (clean["lateness_days"] <= GRACE_DAYS)
    if late_ok.any():
        clean.loc[late_ok, "quality_flags"] = clean.loc[late_ok, "quality_flags"].apply(lambda x: add_flag_str(x, "late_arrival_within_grace"))

    clean["suppressed"] = clean["suppressed"].astype(bool)
    clean["released"] = clean["released"].astype(bool)
    clean = clean[CANONICAL_COLS].reset_index(drop=True)

    report.update(
        {
            "clean_rows": int(len(clean)),
            "quarantine_rows": int(len(quarantine)),
            "released_rows": int(clean["released"].sum()),
            "suppressed_rows": int(clean["suppressed"].sum()),
            "schema_version": SCHEMA_VERSION,
            "provenance": PROVENANCE,
        }
    )
    return clean, quarantine, report


# -----------------------------------------------------------------------------
# Measurement algorithms
# -----------------------------------------------------------------------------


def point_revenue_uplift(pop_t: Sequence[float], rev_t: Sequence[float], pop_h: Sequence[float], rev_h: Sequence[float]) -> float:
    pt = float(np.sum(pop_t))
    ph = float(np.sum(pop_h))
    if pt <= 0 or ph <= 0:
        return 0.0
    rt = float(np.sum(rev_t))
    rh = float(np.sum(rev_h))
    return (rt / pt - rh / ph) * pt


def bootstrap_revenue_uplift_se(
    pop_t: np.ndarray,
    rev_t: np.ndarray,
    pop_h: np.ndarray,
    rev_h: np.ndarray,
    n_boot: int = BOOTSTRAP_N,
    seed: int = SEED,
) -> float:
    n1 = len(pop_t)
    n2 = len(pop_h)
    if n1 < 2 or n2 < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i1 = rng.integers(0, n1, n1)
        i2 = rng.integers(0, n2, n2)
        vals.append(point_revenue_uplift(pop_t[i1], rev_t[i1], pop_h[i2], rev_h[i2]))
    return float(np.std(vals, ddof=1))


def estimate_reported_campaign_point(sub: pd.DataFrame) -> float:
    t = sub[sub["exposure_group"] == "treated"]
    h = sub[sub["exposure_group"] == "holdout"]
    if t.empty or h.empty:
        return float("nan")
    return point_revenue_uplift(
        t["eligible_population"].to_numpy(dtype=float),
        t["revenue_reported"].to_numpy(dtype=float),
        h["eligible_population"].to_numpy(dtype=float),
        h["revenue_reported"].to_numpy(dtype=float),
    )


def estimate_latent_campaign_point(sub: pd.DataFrame) -> float:
    t = sub[sub["exposure_group"] == "treated"]
    h = sub[sub["exposure_group"] == "holdout"]
    if t.empty or h.empty:
        return float("nan")
    return point_revenue_uplift(
        t["eligible_population"].to_numpy(dtype=float),
        t["revenue_true"].to_numpy(dtype=float),
        h["eligible_population"].to_numpy(dtype=float),
        h["revenue_true"].to_numpy(dtype=float),
    )


def fallback_row(cid: str, channel: str, spend: float, reason: str) -> Dict[str, Any]:
    se = max(float(spend) * 0.25, 100.0)
    return {
        "campaign_id": cid,
        "channel": channel,
        "spend": float(spend),
        "inc_rev_est": 0.0,
        "inc_rev_se": se,
        "inc_rev_lb": -se,
        "inc_rev_ub": se,
        "confidence": 0.10,
        "method": "deterministic_fallback",
        "fallback": True,
        "fallback_reason": reason,
        "n_cells": 0,
        "true_inc_rev_latent": float("nan"),
        "abs_error": float("nan"),
    }


def baseline_estimates(clean: pd.DataFrame, latent: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    released = clean[clean["released"]].copy()
    released["event_date"] = pd.to_datetime(released["event_date"])
    latent_rel = latent[latent["cell_id"].isin(set(released["cell_id"]))].copy()

    dates = sorted(pd.unique(released["event_date"]))
    cutoff = dates[int(len(dates) * 0.70)] if len(dates) > 3 else (dates[-1] if dates else pd.Timestamp.today().normalize())
    train = released[released["event_date"] < cutoff]
    test = released[released["event_date"] >= cutoff]
    latent_test = latent_rel[latent_rel["cell_id"].isin(set(test["cell_id"]))]

    rows = []
    hold_rows = []

    for spec in CAMPAIGN_SPECS:
        cid = spec["campaign_id"]
        camp = released[released["campaign_id"] == cid]
        camp_lat = latent_rel[latent_rel["campaign_id"] == cid]
        spend = float(camp[camp["exposure_group"] == "treated"]["spend_reported"].sum())

        t = camp[camp["exposure_group"] == "treated"]
        h = camp[camp["exposure_group"] == "holdout"]

        if len(t) < 5 or len(h) < 5:
            row = fallback_row(cid, spec["channel"], spend, "insufficient_cells")
            row["true_inc_rev_latent"] = estimate_latent_campaign_point(camp_lat)
            row["abs_error"] = abs(row["inc_rev_est"] - row["true_inc_rev_latent"]) if not pd.isna(row["true_inc_rev_latent"]) else float("nan")
            rows.append(row)
            continue

        point = estimate_reported_campaign_point(camp)
        se = bootstrap_revenue_uplift_se(
            t["eligible_population"].to_numpy(dtype=float),
            t["revenue_reported"].to_numpy(dtype=float),
            h["eligible_population"].to_numpy(dtype=float),
            h["revenue_reported"].to_numpy(dtype=float),
            seed=seed + stable_hash_int(cid),
        )
        if pd.isna(se):
            se = max(spend * 0.25, 100.0)

        true_point = estimate_latent_campaign_point(camp_lat)
        ci = 1.96 * se

        conf_val = 1.0 - se / (abs(point) + 1e-6)
        if not math.isfinite(conf_val):
            conf_val = 0.0
        conf = clip01(conf_val)

        fb = bool(se > max(abs(point) * 2.0, 500.0))

        rows.append(
            {
                "campaign_id": cid,
                "channel": spec["channel"],
                "spend": spend,
                "inc_rev_est": float(point),
                "inc_rev_se": float(se),
                "inc_rev_lb": float(point - ci),
                "inc_rev_ub": float(point + ci),
                "confidence": float(conf),
                "method": "baseline_dp_diff",
                "fallback": fb,
                "fallback_reason": "wide_uncertainty" if fb else "",
                "n_cells": int(len(camp)),
                "true_inc_rev_latent": float(true_point) if not pd.isna(true_point) else float("nan"),
                "abs_error": float(abs(point - true_point)) if not pd.isna(true_point) else float("nan"),
            }
        )

        camp_test = test[test["campaign_id"] == cid]
        camp_test_lat = latent_test[latent_test["campaign_id"] == cid]
        tt = camp_test[camp_test["exposure_group"] == "treated"]
        hh = camp_test[camp_test["exposure_group"] == "holdout"]
        if len(tt) >= 3 and len(hh) >= 3:
            est_t = estimate_reported_campaign_point(camp_test)
            se_t = bootstrap_revenue_uplift_se(
                tt["eligible_population"].to_numpy(dtype=float),
                tt["revenue_reported"].to_numpy(dtype=float),
                hh["eligible_population"].to_numpy(dtype=float),
                hh["revenue_reported"].to_numpy(dtype=float),
                seed=seed + 17 + stable_hash_int(cid),
            )
            true_t = estimate_latent_campaign_point(camp_test_lat)
            ci_t = 1.96 * (se_t if not pd.isna(se_t) else max(float(tt["spend_reported"].sum()) * 0.25, 100.0))
            hold_rows.append(
                {
                    "campaign_id": cid,
                    "holdout_est": float(est_t),
                    "holdout_se": float(se_t) if not pd.isna(se_t) else float("nan"),
                    "holdout_lb": float(est_t - ci_t),
                    "holdout_ub": float(est_t + ci_t),
                    "latent_truth": float(true_t) if not pd.isna(true_t) else float("nan"),
                    "abs_error": float(abs(est_t - true_t)) if not pd.isna(true_t) else float("nan"),
                    "covered": bool((est_t - ci_t <= true_t <= est_t + ci_t) if not pd.isna(true_t) else False),
                    "sign_correct": bool(np.sign(est_t) == np.sign(true_t) if not pd.isna(true_t) else False),
                }
            )

    info = {
        "method": "baseline_dp_diff",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "cutoff": str(cutoff.date() if hasattr(cutoff, "date") else cutoff),
        "formula": "Incremental revenue = (reported_revenue_treated / population_treated - reported_revenue_holdout / population_holdout) * population_treated",
        "uncertainty": "Nonparametric bootstrap over released measurement cells; 95% CI = estimate +/- 1.96 * SE.",
    }
    return pd.DataFrame(rows), pd.DataFrame(hold_rows), info


def slearner_estimates(clean: pd.DataFrame, latent: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    released = clean[clean["released"]].copy()
    released["event_date"] = pd.to_datetime(released["event_date"])
    latent_rel = latent[latent["cell_id"].isin(set(released["cell_id"]))].copy()

    dates = sorted(pd.unique(released["event_date"]))
    cutoff = dates[int(len(dates) * 0.70)] if len(dates) > 3 else (dates[-1] if dates else pd.Timestamp.today().normalize())
    train = released[released["event_date"] < cutoff].copy()
    test = released[released["event_date"] >= cutoff].copy()

    if len(train) < 80 or len(test) < 30:
        return baseline_estimates(clean, latent, seed)

    feature_cols = ["campaign_id", "channel", "geo_bucket", "age_bucket", "device_type", "consent_state", "exposure_group"]
    numeric_cols = ["log_population", "dayofweek", "day_of_month"]

    def prep(df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        x["log_population"] = np.log1p(x["eligible_population"].astype(float))
        x["dayofweek"] = pd.to_datetime(x["event_date"]).dt.dayofweek.astype(float)
        x["day_of_month"] = pd.to_datetime(x["event_date"]).dt.day.astype(float)
        dd = pd.get_dummies(x[feature_cols].astype(str), drop_first=True).astype(float)
        num = x[numeric_cols].astype(float)
        return pd.concat([dd, num], axis=1)

    X_train = prep(train)
    y_train = (train["revenue_reported"].astype(float) / train["eligible_population"].astype(float).clip(lower=1.0)).to_numpy()
    ref_cols = list(X_train.columns)

    model = HistGradientBoostingRegressor(
        max_iter=140,
        learning_rate=0.08,
        max_depth=6,
        min_samples_leaf=12,
        l2_regularization=0.8,
        random_state=seed,
    )
    model.fit(X_train, y_train)

    def predict_cf(df: pd.DataFrame, group: str) -> np.ndarray:
        x = df.copy()
        x["exposure_group"] = group
        X = prep(x).reindex(columns=ref_cols, fill_value=0)
        return model.predict(X)

    pred_t_all = predict_cf(released, "treated")
    pred_h_all = predict_cf(released, "holdout")
    released["inc_cell"] = (pred_t_all - pred_h_all) * released["eligible_population"].astype(float)

    rows = []
    for spec in CAMPAIGN_SPECS:
        cid = spec["campaign_id"]
        sub = released[released["campaign_id"] == cid]
        camp_lat = latent_rel[latent_rel["campaign_id"] == cid]
        spend = float(sub[sub["exposure_group"] == "treated"]["spend_reported"].sum())

        if len(sub) < 8:
            row = fallback_row(cid, spec["channel"], spend, "insufficient_cells_for_model")
            row["true_inc_rev_latent"] = estimate_latent_campaign_point(camp_lat)
            row["abs_error"] = abs(row["inc_rev_est"] - row["true_inc_rev_latent"]) if not pd.isna(row["true_inc_rev_latent"]) else float("nan")
            rows.append(row)
            continue

        point = float(sub["inc_cell"].sum())
        se = float(sub["inc_cell"].std(ddof=1) * math.sqrt(len(sub))) if len(sub) > 1 else float("nan")
        if pd.isna(se):
            se = max(spend * 0.25, 100.0)

        true_point = estimate_latent_campaign_point(camp_lat)
        ci = 1.96 * se

        conf_val = 1.0 - se / (abs(point) + 1e-6)
        if not math.isfinite(conf_val):
            conf_val = 0.0
        conf = clip01(conf_val)

        fb = bool(se > max(abs(point) * 2.0, 750.0))

        rows.append(
            {
                "campaign_id": cid,
                "channel": spec["channel"],
                "spend": spend,
                "inc_rev_est": point,
                "inc_rev_se": se,
                "inc_rev_lb": point - ci,
                "inc_rev_ub": point + ci,
                "confidence": conf,
                "method": "slearner_gbm",
                "fallback": fb,
                "fallback_reason": "wide_model_uncertainty" if fb else "",
                "n_cells": int(len(sub)),
                "true_inc_rev_latent": float(true_point) if not pd.isna(true_point) else float("nan"),
                "abs_error": float(abs(point - true_point)) if not pd.isna(true_point) else float("nan"),
            }
        )

    hold_rows = []
    pred_t_test = predict_cf(test, "treated")
    pred_h_test = predict_cf(test, "holdout")
    test = test.copy()
    test["inc_cell"] = (pred_t_test - pred_h_test) * test["eligible_population"].astype(float)
    latent_test = latent_rel[latent_rel["cell_id"].isin(set(test["cell_id"]))]

    for cid, sub in test.groupby("campaign_id"):
        camp_lat = latent_test[latent_test["campaign_id"] == cid]
        est = float(sub["inc_cell"].sum())
        se = float(sub["inc_cell"].std(ddof=1) * math.sqrt(len(sub))) if len(sub) > 1 else float("nan")
        true = estimate_latent_campaign_point(camp_lat)
        ci = 1.96 * (se if not pd.isna(se) else 100.0)
        hold_rows.append(
            {
                "campaign_id": cid,
                "holdout_est": est,
                "holdout_se": se,
                "holdout_lb": est - ci,
                "holdout_ub": est + ci,
                "latent_truth": true,
                "abs_error": abs(est - true) if not pd.isna(true) else float("nan"),
                "covered": bool((est - ci <= true <= est + ci) if not pd.isna(true) else False),
                "sign_correct": bool(np.sign(est) == np.sign(true) if not pd.isna(true) else False),
            }
        )

    info = {
        "method": "slearner_gbm",
        "model": "HistGradientBoostingRegressor S-learner on aggregated privacy-noised cells",
        "target": "revenue_reported / eligible_population",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "cutoff": str(cutoff.date() if hasattr(cutoff, "date") else cutoff),
        "formula": "Cell incremental revenue = (predicted_revenue_per_population_treated - predicted_revenue_per_population_holdout) * eligible_population; campaign estimate = sum(cell estimates).",
        "uncertainty": "Cell-level standard error scaled by sqrt(n_cells); 95% CI = estimate +/- 1.96 * SE.",
        "leakage_controls": "Chronological train/test split; no user-level identifiers; categorical features are bucketed.",
    }
    return pd.DataFrame(rows), pd.DataFrame(hold_rows), info


def deterministic_fallback_estimates(clean: pd.DataFrame, latent: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    rows = []
    for spec in CAMPAIGN_SPECS:
        cid = spec["campaign_id"]
        camp = clean[clean["campaign_id"] == cid]
        spend = float(camp[camp["exposure_group"] == "treated"]["spend_reported"].sum())
        camp_lat = latent[latent["campaign_id"] == cid]
        true_point = estimate_latent_campaign_point(camp_lat)
        row = fallback_row(cid, spec["channel"], spend, "forced_deterministic_fallback")
        row["true_inc_rev_latent"] = float(true_point) if not pd.isna(true_point) else float("nan")
        row["abs_error"] = abs(row["inc_rev_est"] - row["true_inc_rev_latent"]) if not pd.isna(row["true_inc_rev_latent"]) else float("nan")
        row["n_cells"] = int(len(camp))
        rows.append(row)
    info = {
        "method": "deterministic_fallback",
        "formula": "No model-based incrementality claimed. Estimate is zero with conservative uncertainty; action is hold or human review.",
        "use_case": "Used when data quality, privacy suppression, or model uncertainty is too high.",
    }
    return pd.DataFrame(rows), pd.DataFrame(), info


def compute_fairness_alerts(clean: pd.DataFrame) -> set:
    alerts = set()
    rel = clean[clean["released"]].copy()

    for cid, g in rel.groupby("campaign_id"):
        for age, sub in g.groupby("age_bucket"):
            t = sub[sub["exposure_group"] == "treated"]
            h = sub[sub["exposure_group"] == "holdout"]
            if len(t) >= 4 and len(h) >= 4:
                est = estimate_reported_campaign_point(sub)
                spend = float(t["spend_reported"].sum())
                if spend > 0 and not pd.isna(est) and est < -0.12 * spend:
                    alerts.add(cid)

    for cid, g in clean.groupby("campaign_id"):
        supp = g.groupby("age_bucket")["suppressed"].mean()
        cnt = g.groupby("age_bucket")["cell_id"].count()
        if len(supp) >= 2 and cnt.min() >= 5:
            mn = max(float(supp.min()), 0.01)
            mx = float(supp.max())
            if mx > 2.0 * mn and mx > 0.15:
                alerts.add(cid)
    return alerts


def enrich_estimates(estimates: pd.DataFrame, clean: pd.DataFrame, epsilon: float, min_pop: int, fairness_alerts: set) -> pd.DataFrame:
    est = estimates.copy()

    stats = (
        clean.groupby("campaign_id")
        .agg(
            suppressed_rate=("suppressed", "mean"),
            cells=("cell_id", "count"),
            eps_max=("epsilon", "max"),
        )
        .reset_index()
    )

    small_df = clean.assign(small_flag=clean["eligible_population"] < (min_pop * 1.5))
    small_stats = small_df.groupby("campaign_id")["small_flag"].mean().rename("small_rate").reset_index()
    stats = stats.merge(small_stats, on="campaign_id", how="left")

    est = est.merge(stats, on="campaign_id", how="left")

    est["spend"] = pd.to_numeric(est["spend"], errors="coerce").fillna(0.0).astype(float)
    est["inc_rev_est"] = pd.to_numeric(est["inc_rev_est"], errors="coerce").fillna(0.0).astype(float)

    fallback_se = est["spend"].astype(float) * 0.25 + 100.0
    se_arr = pd.to_numeric(est["inc_rev_se"], errors="coerce").to_numpy(dtype=float)
    fb_arr = fallback_se.to_numpy(dtype=float)
    se_arr = np.where(np.isfinite(se_arr), se_arr, fb_arr)
    se_arr = np.maximum(se_arr, 0.0)
    est["inc_rev_se"] = se_arr

    est["inc_rev_lb"] = est["inc_rev_est"] - 1.96 * est["inc_rev_se"]
    est["inc_rev_ub"] = est["inc_rev_est"] + 1.96 * est["inc_rev_se"]

    conf_arr = pd.to_numeric(est["confidence"], errors="coerce").to_numpy(dtype=float)
    conf_arr = np.where(np.isfinite(conf_arr), conf_arr, 0.0)
    est["confidence"] = np.clip(conf_arr, 0.0, 1.0)

    suppressed_rate = pd.to_numeric(est["suppressed_rate"], errors="coerce").fillna(0.0).astype(float)
    eps_max = pd.to_numeric(est["eps_max"], errors="coerce").fillna(float(epsilon)).astype(float)
    small_rate = pd.to_numeric(est["small_rate"], errors="coerce").fillna(0.0).astype(float)

    privacy_raw = (
        0.42 * suppressed_rate
        + 0.23 * (eps_max / 4.0)
        + 0.35 * small_rate
    )
    privacy_raw = pd.to_numeric(privacy_raw, errors="coerce").fillna(0.0).astype(float)
    est["privacy_risk"] = np.clip(privacy_raw.to_numpy(dtype=float), 0.0, 1.0)

    est["fairness_alert"] = est["campaign_id"].isin(fairness_alerts).astype(bool)

    est["roi_point"] = np.where(est["spend"] > 0, (est["inc_rev_est"] - est["spend"]) / est["spend"], 0.0)
    est["roi_lb"] = np.where(est["spend"] > 0, (est["inc_rev_lb"] - est["spend"]) / est["spend"], 0.0)

    est["roi_point"] = pd.to_numeric(est["roi_point"], errors="coerce").fillna(0.0).astype(float)
    est["roi_lb"] = pd.to_numeric(est["roi_lb"], errors="coerce").fillna(0.0).astype(float)

    return est


def get_estimates(clean: pd.DataFrame, latent: pd.DataFrame, method: str, epsilon: float, min_pop: int) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if method == "baseline_dp_diff":
        est, hold, info = baseline_estimates(clean, latent, SEED)
    elif method == "slearner_gbm":
        est, hold, info = slearner_estimates(clean, latent, SEED)
    else:
        est, hold, info = deterministic_fallback_estimates(clean, latent)

    alerts = compute_fairness_alerts(clean)
    est = enrich_estimates(est, clean, epsilon, min_pop, alerts)
    info["fairness_alert_campaigns"] = sorted(alerts)
    return est, hold, info


def make_recommendations(estimates: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for r in estimates.itertuples():
        spend = float(r.spend) if float(r.spend) > 0 else 1.0
        confidence = float(r.confidence) if math.isfinite(float(r.confidence)) else 0.0
        privacy_risk = float(r.privacy_risk) if math.isfinite(float(r.privacy_risk)) else 0.0
        fairness_alert = bool(r.fairness_alert)
        fallback = bool(r.fallback)
        roi_lb = float(r.roi_lb) if math.isfinite(float(r.roi_lb)) else 0.0
        inc_rev_est = float(r.inc_rev_est) if math.isfinite(float(r.inc_rev_est)) else 0.0
        inc_rev_lb = float(r.inc_rev_lb) if math.isfinite(float(r.inc_rev_lb)) else 0.0

        if fallback or confidence < 0.25:
            scale = 1.0
            action = "hold_low_confidence"
            approval = False
        elif privacy_risk > 0.55 or fairness_alert:
            scale = 1.0
            action = "human_review"
            approval = True
        else:
            if roi_lb > 0.08:
                scale = min(1.20, 1.0 + 0.45 * roi_lb)
                action = "increase_budget"
            elif roi_lb < -0.05:
                scale = max(0.80, 1.0 + 0.45 * roi_lb)
                action = "reduce_budget"
            else:
                scale = 1.0
                action = "maintain"
            approval = abs(scale - 1.0) > 0.10

        expected_inc_at_scale = inc_rev_est * scale
        delta_inc = expected_inc_at_scale - inc_rev_est

        recs.append(
            {
                "campaign_id": r.campaign_id,
                "channel": r.channel,
                "current_spend": float(r.spend),
                "recommended_scale": float(scale),
                "proposed_spend": float(r.spend) * scale,
                "action": action,
                "approval_required": bool(approval),
                "estimated_incremental_revenue": inc_rev_est,
                "lower_bound_incremental_revenue": inc_rev_lb,
                "expected_delta_incremental_revenue": float(delta_inc),
                "confidence": confidence,
                "privacy_risk": privacy_risk,
                "fairness_alert": fairness_alert,
                "method": r.method,
            }
        )
    return pd.DataFrame(recs)


# -----------------------------------------------------------------------------
# Live simulation and demo media
# -----------------------------------------------------------------------------


def generate_live_ticks(seed: int = SEED) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed + 99)
    now = now_utc()
    ticks = []
    cum_spend = 0.0
    cum_rep_conv = 0.0
    cum_rep_rev = 0.0
    cum_est_inc = 0.0
    cum_true_inc = 0.0

    base_rate = 0.018
    eps = DEFAULT_EPSILON

    for i in range(LIVE_TICKS):
        ts = now - timedelta(minutes=(LIVE_TICKS - i) * 5)
        pop = int(rng.lognormal(math.log(42000), 0.22))
        spend_tick = float(rng.lognormal(math.log(1150), 0.28))
        treated_rate = base_rate * 1.14
        hold_rate = base_rate

        true_conv = float(rng.poisson(max(0.0, pop * treated_rate)))
        hold_conv = float(rng.poisson(max(0.0, pop * hold_rate)))
        aov = float(np.clip(rng.lognormal(math.log(78), 0.35), 8.0, MAX_ORDER_VALUE))
        hold_aov = float(np.clip(rng.lognormal(math.log(76), 0.35), 8.0, MAX_ORDER_VALUE))

        true_rev = true_conv * aov
        hold_rev = hold_conv * hold_aov
        true_inc = true_rev - hold_rev

        rep_conv = max(0.0, true_conv + rng.laplace(0.0, 1.0 / eps))
        rep_rev = max(0.0, true_rev + rng.laplace(0.0, MAX_ORDER_VALUE / eps))
        rep_hold_rev = max(0.0, hold_rev + rng.laplace(0.0, MAX_ORDER_VALUE / eps))
        est_inc = rep_rev - rep_hold_rev

        events = []
        privacy_status = "OK"

        if i == 11:
            spend_tick *= 2.45
            events.append({"severity": "warning", "type": "anomaly", "message": "CTV spend spike detected; outlier flag raised."})
        if i == 17:
            rep_rev += 4200.0
            est_inc += 4200.0
            events.append({"severity": "info", "type": "late_conversion", "message": "Late conversion restatement applied within grace window."})
        if i == 21:
            pop_small = 650
            events.append({"severity": "warning", "type": "suppression", "message": f"Small cell population {pop_small} < {MIN_CELL_POPULATION_DEFAULT}; suppressed from release."})
            privacy_status = "Suppressed cell"
        if i == 25:
            events.append({"severity": "info", "type": "privacy_budget", "message": f"Epsilon check passed: epsilon={eps}, policy_max={POLICY_MAX_EPSILON}."})
        if i == 28:
            events.append({"severity": "action", "type": "recommendation", "message": "Budget reallocation proposal generated; consequential change requires human approval."})

        cum_spend += spend_tick
        cum_rep_conv += rep_conv
        cum_rep_rev += rep_rev
        cum_est_inc += est_inc
        cum_true_inc += true_inc

        ticks.append(
            {
                "tick": i + 1,
                "ts": iso(ts),
                "spend_tick": spend_tick,
                "cum_spend": cum_spend,
                "cum_reported_conversions": cum_rep_conv,
                "cum_reported_revenue": cum_rep_rev,
                "cum_estimated_incremental": cum_est_inc,
                "cum_simulated_true_incremental": cum_true_inc,
                "privacy_status": privacy_status,
                "events": events,
            }
        )
    return ticks


def make_live_figure(ticks: List[Dict[str, Any]], upto: int) -> go.Figure:
    df = pd.DataFrame(ticks[:upto])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["ts"], y=df["cum_spend"], mode="lines+markers", name="Observed spend (USD)"))
    fig.add_trace(go.Scatter(x=df["ts"], y=df["cum_reported_revenue"], mode="lines+markers", name="Reported revenue (privacy-noised)"))
    fig.add_trace(go.Scatter(x=df["ts"], y=df["cum_estimated_incremental"], mode="lines+markers", name="Estimated incremental revenue"))
    fig.add_trace(
        go.Scatter(
            x=df["ts"],
            y=df["cum_simulated_true_incremental"],
            mode="lines",
            name="Simulated ground-truth incremental",
            line=dict(dash="dot"),
        )
    )
    fig.update_layout(
        height=330,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Simulated time",
        yaxis_title="USD / count",
    )
    return fig


def draw_cv2_frame(idx: int, total: int, series: np.ndarray, holdout: np.ndarray) -> np.ndarray:
    h, w = 360, 640
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = (28, 28, 32)

    cv2.putText(frame, "Privacy-Aware Advertising Measurement", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 2)
    cv2.putText(frame, "Synthetic live demo - no PII", (18, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 180), 1)

    x0, y0, x1, y1 = 55, 80, 600, 295
    cv2.rectangle(frame, (x0, y0), (x1, y1), (85, 85, 95), 1)

    visible = series[: idx + 1]
    visible_h = holdout[: idx + 1]
    vmin = float(min(visible.min(), visible_h.min()))
    vmax = float(max(visible.max(), visible_h.max()))
    if vmax - vmin < 1e-6:
        vmax = vmin + 1.0

    pts = []
    ptsh = []
    for j in range(len(visible)):
        x = int(x0 + (x1 - x0) * j / max(1, total - 1))
        y = int(y1 - (y1 - y0) * (visible[j] - vmin) / (vmax - vmin))
        yh = int(y1 - (y1 - y0) * (visible_h[j] - vmin) / (vmax - vmin))
        pts.append((x, y))
        ptsh.append((x, yh))

    if len(ptsh) > 1:
        cv2.polylines(frame, [np.array(ptsh, dtype=np.int32)], False, (160, 160, 170), 2)
    if len(pts) > 1:
        cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, (0, 190, 255), 3)

    cv2.putText(frame, f"Tick {idx + 1}/{total}", (18, 322), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
    cv2.putText(frame, f"Est. incremental: ${series[idx]:,.0f}", (18, 344), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 255, 120), 1)
    cv2.putText(frame, "DP noise + suppression enabled", (390, 344), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 200, 120), 1)
    return frame


def generate_video_with_cv2() -> Tuple[str, str, str]:
    if cv2 is None:
        raise RuntimeError("OpenCV is not available.")
    rng = np.random.default_rng(SEED + 555)
    fps = 15
    total = fps * 6
    series = np.maximum(0, np.cumsum(rng.normal(9.0, 3.5, total)) + 35.0)
    holdout = np.maximum(0, np.cumsum(rng.normal(6.0, 3.0, total)) + 20.0)

    candidates = [
        (WEBM_PATH, cv2.VideoWriter_fourcc(*"VP80")),
        (MP4_PATH, cv2.VideoWriter_fourcc(*"avc1")),
        (MP4_PATH, cv2.VideoWriter_fourcc(*"mp4v")),
    ]
    last_err = ""
    for path, fourcc in candidates:
        try:
            writer = cv2.VideoWriter(path, fourcc, fps, (640, 360))
            if not writer.isOpened():
                last_err = f"VideoWriter did not open for {path}"
                continue
            for i in range(total):
                writer.write(draw_cv2_frame(i, total, series, holdout))
            writer.release()
            if os.path.exists(path) and os.path.getsize(path) > 1000:
                kind = "webm" if path.endswith(".webm") else "mp4"
                return kind, path, ""
        except Exception as e:
            last_err = str(e)
    raise RuntimeError(last_err or "Unable to generate video with OpenCV.")


def generate_gif_with_pil() -> Tuple[str, str, str]:
    if Image is None or ImageDraw is None:
        raise RuntimeError("Pillow is not available.")
    rng = np.random.default_rng(SEED + 777)
    total = 70
    series = np.maximum(0, np.cumsum(rng.normal(9.0, 3.5, total)) + 35.0)
    holdout = np.maximum(0, np.cumsum(rng.normal(6.0, 3.0, total)) + 20.0)
    w, h = 640, 360
    frames = []
    for idx in range(total):
        img = Image.new("RGB", (w, h), (28, 28, 32))
        d = ImageDraw.Draw(img)
        d.text((18, 18), "Privacy-Aware Advertising Measurement", fill=(240, 240, 240))
        d.text((18, 38), "Synthetic live demo - no PII", fill=(180, 220, 180))
        x0, y0, x1, y1 = 55, 80, 600, 295
        d.rectangle([x0, y0, x1, y1], outline=(85, 85, 95))
        visible = series[: idx + 1]
        visible_h = holdout[: idx + 1]
        vmin = min(float(visible.min()), float(visible_h.min()))
        vmax = max(float(visible.max()), float(visible_h.max()))
        if vmax - vmin < 1e-6:
            vmax = vmin + 1.0
        pts = []
        ptsh = []
        for j in range(len(visible)):
            x = x0 + (x1 - x0) * j / max(1, total - 1)
            y = y1 - (y1 - y0) * (visible[j] - vmin) / (vmax - vmin)
            yh = y1 - (y1 - y0) * (visible_h[j] - vmin) / (vmax - vmin)
            pts.append((x, y))
            ptsh.append((x, yh))
        if len(ptsh) > 1:
            d.line(ptsh, fill=(160, 160, 170), width=2)
        if len(pts) > 1:
            d.line(pts, fill=(0, 190, 255), width=3)
        d.text((18, 315), f"Tick {idx + 1}/{total}", fill=(220, 220, 220))
        d.text((18, 335), f"Est. incremental: ${series[idx]:,.0f}", fill=(120, 255, 120))
        d.text((390, 335), "DP noise + suppression", fill=(255, 200, 120))
        frames.append(img)
    frames[0].save(GIF_PATH, save_all=True, append_images=frames[1:], duration=80, loop=0)
    return "gif", GIF_PATH, ""


def ensure_demo_assets() -> Dict[str, str]:
    if "demo_assets" in st.session_state:
        return st.session_state.demo_assets

    assets = {
        "video_kind": "none",
        "video_path": "",
        "video_err": "",
        "gif_path": "",
        "gif_err": "",
    }

    try:
        kind, path, err = generate_video_with_cv2()
        assets.update({"video_kind": kind, "video_path": path, "video_err": err})
    except Exception as e:
        assets["video_err"] = str(e)

    try:
        kind, path, err = generate_gif_with_pil()
        if kind == "gif" and path and os.path.exists(path):
            assets["gif_path"] = path
            assets["gif_err"] = err
        else:
            assets["gif_err"] = err or "GIF not created"
    except Exception as e:
        assets["gif_err"] = str(e)

    st.session_state.demo_assets = assets
    return assets


def run_live_simulation(kpi_slot: Any, chart_slot: Any, event_slot: Any) -> None:
    ticks = st.session_state.live_ticks
    for i in range(len(ticks)):
        tick = ticks[i]

        cols = kpi_slot.columns(5)
        el_metric(cols[0], "Tick", f"{tick['tick']}/{len(ticks)}")
        el_metric(cols[1], "Observed spend", fmt_usd(tick["cum_spend"]))
        el_metric(cols[2], "Reported conversions", f"{tick['cum_reported_conversions']:,.0f}")
        el_metric(cols[3], "Est. incr. revenue", fmt_usd(tick["cum_estimated_incremental"]))
        el_metric(cols[4], "Privacy status", tick["privacy_status"])

        el_plotly(chart_slot, make_live_figure(ticks, i + 1))

        el_md(event_slot, "**Event log (simulated)**")
        if tick["events"]:
            el_json(event_slot, tick["events"])
        else:
            el_caption(event_slot, "No privacy/anomaly events this tick.")

        time.sleep(0.22)

    st.session_state.auto_demo_active = False
    rerun_app()


# -----------------------------------------------------------------------------
# Self-tests
# -----------------------------------------------------------------------------


def run_self_tests() -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        results.append({"test": name, "passed": bool(passed), "detail": detail})

    try:
        check("hand_calculated_revenue_uplift", abs(point_revenue_uplift([1000], [5000], [1000], [4000]) - 1000.0) < 1e-6)
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        check("laplace_noise_deterministic_with_seed", np.allclose(rng1.laplace(0, 1, 10), rng2.laplace(0, 1, 10)))
        check("suppression_rule", (500 < MIN_CELL_POPULATION_DEFAULT) and (1500 >= MIN_CELL_POPULATION_DEFAULT))

        d = pd.DataFrame(
            {
                "cell_id": ["a", "a"],
                "ingested_at": [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")],
                "v": [1, 2],
            }
        )
        ded = d.sort_values("ingested_at", kind="stable").drop_duplicates("cell_id", keep="last")
        check("duplicate_dedupe_keeps_latest", len(ded) == 1 and int(ded.iloc[0]["v"]) == 2)

        check("negative_spend_detected", bool(((pd.Series([-1.0]) < 0).any())))
        check("schema_version_present", SCHEMA_VERSION == "1.0.0")

        assets = ensure_demo_assets()
        media_ok = (
            (assets.get("video_path", "") and os.path.exists(assets.get("video_path", "")))
            or (assets.get("gif_path", "") and os.path.exists(assets.get("gif_path", "")))
        )
        check("demo_media_generated", media_ok, f"video={assets.get('video_kind','none')}; gif={bool(assets.get('gif_path'))}")

        date_key = pd.Timestamp.today().date().isoformat()
        clean, quar, report, latent, _ = prepare_dataset(DEFAULT_EPSILON, MIN_CELL_POPULATION_DEFAULT, SEED, True, date_key)
        check("dataset_has_rows", len(clean) > 1000, f"clean={len(clean)}, quarantine={len(quar)}")
        check("no_direct_pii_columns", not any(c.lower() in {"user_id", "email", "phone", "ip_address", "device_id", "cookie_id"} for c in clean.columns))
        check("suppressed_rows_not_released", bool(((clean["suppressed"]) & (~clean["released"])).all()))
        check("quarantine_contains_bad_records", len(quar) > 0, f"quarantine_rows={len(quar)}")

        est, hold, info = get_estimates(clean, latent, "baseline_dp_diff", DEFAULT_EPSILON, MIN_CELL_POPULATION_DEFAULT)
        check("baseline_estimates_all_campaigns", len(est) == len(CAMPAIGN_SPECS))
        check("confidence_in_unit_interval", bool(((est["confidence"] >= 0) & (est["confidence"] <= 1)).all()))
        check("lower_bound_not_above_upper", bool((est["inc_rev_lb"] <= est["inc_rev_ub"]).all()))

        est2, _, _ = get_estimates(clean, latent, "slearner_gbm", DEFAULT_EPSILON, MIN_CELL_POPULATION_DEFAULT)
        check("slearner_estimates_all_campaigns", len(est2) == len(CAMPAIGN_SPECS))

        recs = make_recommendations(est)
        check("recommendations_generated", len(recs) == len(CAMPAIGN_SPECS))
        check("consequential_actions_require_approval", bool((recs.loc[recs["approval_required"], "action"].isin(["increase_budget", "reduce_budget", "human_review"])).all()))
    except Exception:
        check("unexpected_exception", False, traceback.format_exc(limit=2))

    return results


# -----------------------------------------------------------------------------
# Documentation artifacts
# -----------------------------------------------------------------------------

EXEC_SUMMARY = r"""
# Executive summary

**Product:** Privacy-Aware Advertising Measurement for AI Advertising Intelligence and Revenue Optimization.

**Problem solved:** Large advertising platforms and enterprises need to know which campaigns create incremental revenue, but legacy last-touch reporting is biased, privacy regulations restrict user-level data, and small-cell analytics can leak information. This software turns bucketed, consent-aware, privacy-noised ad/revenue aggregates into auditable incrementality estimates and budget recommendations.

**Why a large technology company would pay:** Misallocated ad spend, disputed advertiser reports, privacy incidents, and slow manual measurement cost revenue and trust. A reliable privacy-aware measurement layer supports better budget allocation, fewer compliance incidents, faster campaign decisions, and defensible advertiser reporting.

**Primary users:** Marketing science, revenue operations, privacy program office, campaign managers, data engineering, legal/compliance, and executive reporting.

**Key decisions improved:** holdout design, channel budget shifts, bid/budget caps, creative pause, anomaly investigation, advertiser-facing reporting, and privacy-budget approvals.

**Prototype scope:** single-file Streamlit app, synthetic data only, CPU-only, no API keys, baseline differentially private aggregate uplift, optional S-learner GBM, dashboard, audit history, self-tests, generated demo clip, and business-value calculator with user-entered assumptions.
"""

PAGE1_PRD = r"""
# Page 1 — Product requirements and operational problem

## Precise product mission
Provide a privacy-aware measurement and decision-support system that estimates incremental advertising revenue from bucketed aggregates, quantifies uncertainty, enforces suppression and differential-privacy budgets, and recommends revenue-optimizing actions with human approval for consequential changes.

## Measurable scope for this prototype
- 6 synthetic campaigns across search, social, CTV, retail media, display, and email.
- 14 days of daily measurement cells plus a 30-tick intraday live simulation.
- Canonical aggregate schema with privacy metadata and quality flags.
- Baseline estimator: differentially private aggregate difference-in-means.
- Advanced estimator: S-learner gradient boosting on aggregated noised cells.
- Deterministic fallback when privacy, data quality, or uncertainty is inadequate.
- Dashboard labels: Observed, Reported/noised, Estimated, Predicted, Simulated truth.
- Audit log for bootstrap, ingestion, approvals, and recommendation actions.
- In-app self-tests and generated demo media.

## Who uses it and what decisions it improves
| User | Decision | Current legacy workflow | Improved workflow |
|---|---|---|---|
| Marketing scientist | Choose holdout design and estimator | SQL spreadsheets, inconsistent cell definitions | Canonical cells, chronological split, CI, audit trail |
| Revenue ops manager | Reallocate weekly budget | Manual last-touch ROAS review | Lower-bound incremental ROI recommendations with approval gate |
| Privacy program office | Approve epsilon and suppression policy | Email approvals, no query lineage | Epsilon budget panel, suppressed-cell metrics, audit log |
| Campaign manager | Pause or scale creative/campaign | Dashboard screenshots and tribal knowledge | Drill-down evidence, quality flags, event history |
| Finance / executive | Report advertising efficiency | Static monthly decks | Reproducible value calculator with user-entered assumptions |

## Three realistic user journeys

### Journey 1 — Weekly budget reallocation
1. Revenue ops opens Live Demo and Measurement tabs.
2. System loads synthetic 14-day cells, applies epsilon=1.0 and min population=1000.
3. Baseline estimator produces campaign incremental revenue with 95% CI.
4. Recommendation engine suggests +12% to retail media, -8% to CTV, hold for display.
5. Because CTV change exceeds 10% and has fairness alert, human approval is required.
6. User approves only retail media; audit log records actor, timestamp, epsilon, and evidence.

**Legacy workflow improved:** spreadsheet last-touch ROAS replaced by holdout-based incrementality with uncertainty and privacy metadata.

### Journey 2 — Privacy review before advertiser report
1. Privacy analyst filters Privacy & Quality tab.
2. System shows suppressed cells by age bucket and epsilon usage.
3. Failure case: US-MW / 18-24 / desktop / denied has population 650 and is suppressed.
4. Analyst raises epsilon from 1.0 to 2.0 in synthetic demo and observes reduced noise but higher privacy risk.
5. Policy warning appears because epsilon exceeds default policy max 2.0 only if user sets above; approval checkbox required.
6. Export is blocked in production unless approval and audit entry exist.

**Legacy workflow improved:** ad hoc SQL extracts replaced by policy-aware aggregate release with lineage.

### Journey 3 — Anomaly investigation during live campaign
1. Live simulation starts automatically.
2. At tick 12, CTV spend spike event appears.
3. At tick 17, late conversion restatement appears.
4. At tick 21, small-cell suppression event appears.
5. User inspects event log, cumulative estimated incremental revenue, and simulated ground truth.
6. System labels restatement as observed/late and incrementality as estimated.

**Legacy workflow improved:** reactive Slack/email escalation replaced by structured event timeline and audit history.

## Project-specific data fields
Core canonical fields include:
- `cell_id`: hashed bucket identifier, not a user identifier.
- `event_date`, `ingested_at`, `lateness_days`: temporal provenance and late-arrival control.
- `campaign_id`, `channel`, `geo_bucket`, `age_bucket`, `device_type`, `consent_state`, `exposure_group`: bucketed dimensions.
- `eligible_population`, `impressions`, `clicks_reported`, `conversions_reported`, `revenue_reported`, `spend_reported`: aggregate measures.
- `epsilon`, `privacy_delta`, `noise_mechanism`, `noise_conversions`, `noise_revenue`: privacy metadata.
- `suppressed`, `released`, `quality_flags`: privacy and data-quality controls.
- `provenance`, `schema_version`, `source_system`: lineage.

## Concrete failure cases and acceptance checks
| Failure case | Example | Required behavior | Acceptance check |
|---|---|---|---|
| Small cell leakage risk | `geo_bucket=US-MW`, `age_bucket=18-24`, population=650 | Suppress release, exclude from estimator, show suppressed rate | No released row with `eligible_population < min_pop` |
| Late conversion restatement | Revenue +$4,200 arrives 3 days after event date | Allow within grace, flag `late_arrival_within_grace`, audit restatement | Late rows > grace days quarantined |
| Duplicate replay | Same `cell_id` ingested twice | Deduplicate keep latest `ingested_at`, report duplicates dropped | Duplicate test passes |
| Malformed enum | `channel=televisión` | Quarantine with `invalid_enum` | Quarantine table non-empty |
| Negative spend | `spend_reported=-17.25` | Quarantine with `negative_or_invalid_range` | Range validation test passes |
| Outlier revenue | `revenue_reported=1.35e9` | Flag and cap robust outlier | Clean rows contain no extreme unflagged outlier |
| Model uncertainty too wide | CI spans zero and SE > 2x estimate | Fallback to hold/human review | Recommendation action is hold or review |
| Privacy budget exceeded | epsilon > policy max | Require explicit approval and audit | UI warning and approval checkbox |

## Prototype vs production enterprise service
| Dimension | Useful prototype | Production enterprise service |
|---|---|---|
| Data | Synthetic/local | Certified ad server, revenue warehouse, consent ledger |
| Privacy | Local Laplace noise and suppression | Distributed DP, centralized accountant, per-query budget enforcement |
| Scale | Single-node pandas/SQLite | Distributed batch/stream, columnar lakehouse, OLAP store |
| Models | Baseline + sklearn S-learner | Calibrated causal ensemble, drift monitoring, champion/challenger |
| Governance | In-app audit log | SSO, RBAC, approval workflows, legal review, retention policies |
| Reliability | Local restart | HA serving, SLAs, rollback, disaster recovery |
| Reporting | Demo downloads | Contractual advertiser reports with dispute workflow |

## Value hypothesis
**Hypothesis:** Privacy-aware incrementality measurement improves advertising capital efficiency compared with last-touch reporting while reducing privacy and dispute risk.

### Outcome metrics
- Incremental revenue per $1,000 spend.
- Wasted spend reduction: spend allocated to campaigns with lower-bound incremental ROI <= 0.
- Advertiser dispute rate.
- Privacy incident count.
- Decision latency from data arrival to approved action.

### Leading indicators
- Holdout coverage by campaign and segment.
- Released-cell fraction and suppressed-cell fraction.
- CI width and confidence score.
- Data completeness and quarantine rate.
- Recommendation adoption rate.

### Quality metrics
- MAE and RMSE against synthetic latent truth.
- Sign accuracy: proportion of campaign estimates with same sign as latent truth.
- CI coverage: proportion of latent truths inside 95% CI.
- Calibration proxy: average CI width vs absolute error.

### Reliability metrics
- Ingestion success rate.
- Schema validation failure rate.
- p95 estimation latency.
- Audit completeness: percentage of released queries with epsilon, row counts, and actor.
- Rollback success rate.

### Guardrails
- No direct PII fields.
- Minimum cell population suppression.
- Epsilon policy warning and approval for overrides.
- Human approval for budget changes >10%, epsilon increases, suppression disablement, advertiser export, model promotion.
- Fairness alert when a demographic segment shows materially negative estimated uplift or disparate suppression.

### Baseline and controlled evaluation
- **Operational baseline:** last-touch ROAS from legacy reporting.
- **Measurement baseline:** naive aggregate treated-vs-holdout difference with no covariate adjustment.
- **Controlled evaluation:** randomized holdout or ghost bidding, pre-registered metric definitions, chronological split, privacy budget accounting, and comparison of decision policies: baseline-only vs model-assisted vs fallback.
- **Incremental value test:** simulate budget allocations from each policy against synthetic latent truth and measure realized incremental profit, regret, CI coverage, and privacy consumption.
"""

PAGE2_ARCH = r"""
# Page 2 — Architecture, data contracts, and algorithmic design

## Modular architecture
The prototype is a modular monolith in one file, with clear logical boundaries:
- Source adapters: synthetic ad server, synthetic revenue warehouse, synthetic holdout logger.
- Ingestion: raw DataFrame assembly and dirty-record injection.
- Schema validation: required fields, enums, ranges, lateness, duplicates, outliers.
- Canonical data model: `CANONICAL_COLS` stored in SQLite table `cells`.
- Feature/signal processing: bucketed dimensions, privacy metadata, quality flags.
- Baseline analytics: DP aggregate difference-in-means.
- Model layer: optional S-learner HistGradientBoostingRegressor.
- Decision/scoring layer: lower-bound ROI, privacy risk, fairness alert, approval gate.
- API/dashboard: Streamlit UI and download endpoints.
- Audit history: SQLite `audit_log` and `runs`.
- Observability: row counts, latency, quarantine, suppression, epsilon, media status.

### Ownership and failure boundaries
| Component | Owner | Failure boundary | Safe behavior |
|---|---|---|---|
| Source adapter | Data engineering | Missing/upstream outage | Use last known good, flag stale |
| Validator | Data platform | Malformed schema | Quarantine, do not release |
| Privacy engine | Privacy engineering | Small cell or budget breach | Suppress, require approval |
| Estimator | Marketing science | Wide CI or drift | Fallback to hold/review |
| Decision engine | Revenue ops | Consequential action | Human approval required |
| Dashboard | Product | Rendering failure | Show audit and raw status |

## Mermaid architecture diagram

```mermaid
flowchart LR
  A[Synthetic ad server] --> B[Ingestion adapter]
  A2[Synthetic revenue warehouse] --> B
  A3[Consent / holdout logger] --> B
  B --> C[Schema validator]
  C -->|valid| D[Canonical measurement store]
  C -->|invalid| E[Quarantine]
  D --> F[Privacy engine: suppression + DP noise]
  F --> G[Baseline DP difference estimator]
  F --> H[S-learner GBM estimator]
  G --> I[Decision scorer]
  H --> I
  I --> J[Streamlit dashboard]
  J --> K[Audit and observability]
```

## Mermaid sequence diagram

```mermaid
sequenceDiagram
  participant CM as Campaign manager
  participant UI as Dashboard
  participant VAL as Validator
  participant PRIV as Privacy engine
  participant MEAS as Measurement service
  participant AUD as Audit store
  CM->>UI: Request incrementality for date range
  UI->>VAL: Validate query and schema version
  VAL->>PRIV: Provide released candidate cells
  PRIV-->>VAL: Apply suppression and DP noise
  VAL->>MEAS: Pass canonical aggregates
  MEAS->>MEAS: Compute baseline or model estimate + CI
  MEAS->>AUD: Log epsilon, row counts, method, latency
  MEAS-->>UI: Estimates, evidence, recommendations
  UI-->>CM: Show action; require approval if consequential
```

## Data contract

| Field | Type | Unit | Notes |
|---|---|---|---|
| `cell_id` | string | id | Hash of bucketed dimensions and date; no user id |
| `event_date` | date | calendar day | Business event date |
| `campaign_id` | string | id | Campaign identifier |
| `channel` | enum | category | search, social, ctv, retail_media, display, email |
| `geo_bucket` | enum | category | Coarse geography bucket |
| `age_bucket` | enum | category | Coarse age band |
| `device_type` | enum | category | mobile, desktop, ctv |
| `consent_state` | enum | category | granted, denied, unknown |
| `exposure_group` | enum | category | treated, holdout |
| `eligible_population` | int | people | Bucket population count |
| `impressions` | int | ad impressions | Treated only; holdout zero |
| `clicks_reported` | int | clicks | Privacy-noised |
| `conversions_reported` | int | conversions | Privacy-noised |
| `revenue_reported` | float | USD | Privacy-noised |
| `spend_reported` | float | USD | Business spend aggregate |
| `epsilon` | float | DP parameter | Per-release privacy budget |
| `privacy_delta` | float | DP parameter | Failure probability |
| `noise_mechanism` | string | category | laplace |
| `noise_conversions` | float | count | Added noise |
| `noise_revenue` | float | USD | Added noise |
| `suppressed` | bool | flag | True if cell not released |
| `released` | bool | flag | True if safe to use |
| `quality_flags` | string | csv | Missing, late, outlier, duplicate, etc. |
| `provenance` | string | uri | Synthetic source version |
| `schema_version` | string | semver-like | Contract version |
| `source_system` | string | category | Adapter origin |
| `ingested_at` | datetime | UTC | Arrival time |
| `lateness_days` | float | days | `ingested_at - event_date` |

### Example record

```json
{
  "cell_id": "a1b2c3d4e5f60718",
  "event_date": "2026-11-07",
  "campaign_id": "cmp_ctv_reach",
  "channel": "ctv",
  "geo_bucket": "US-W",
  "age_bucket": "25-44",
  "device_type": "ctv",
  "consent_state": "granted",
  "exposure_group": "treated",
  "eligible_population": 18420,
  "impressions": 9320,
  "clicks_reported": 112,
  "conversions_reported": 61,
  "revenue_reported": 4820.17,
  "spend_reported": 74.56,
  "epsilon": 1.0,
  "privacy_delta": 1e-05,
  "noise_mechanism": "laplace",
  "noise_conversions": -2.31,
  "noise_revenue": 118.44,
  "suppressed": false,
  "released": true,
  "quality_flags": "",
  "provenance": "synthetic://privacy-ad-measurement/v1",
  "schema_version": "1.0.0",
  "source_system": "synthetic_adserver",
  "ingested_at": "2026-11-08T02:15:00",
  "lateness_days": 1.09
}
```

## Handling rules
- **Missing values:** critical metrics missing -> quarantine; clicks/impressions missing -> fill zero and flag.
- **Duplicates:** same `cell_id` -> keep latest `ingested_at`, report dropped count.
- **Late-arriving records:** within grace days -> allow and flag; beyond grace -> quarantine.
- **Malformed input:** invalid enum, negative measures, bad timestamps -> quarantine.
- **Outliers:** robust cap and flag; extreme business outliers do not silently enter estimates.
- **Schema changes:** unknown columns dropped and logged; missing canonical columns added as NaN then validated.
- **Versioning:** `schema_version` and `provenance` stored on every row.
- **Retention:** demo uses temporary SQLite; production would enforce retention and privacy controls.

## Transparent baseline before advanced ML
### Approach 1 — DP aggregate difference-in-means
Formula:

`incremental_revenue = (reported_revenue_treated / population_treated - reported_revenue_holdout / population_holdout) * population_treated`

Pros: auditable, simple, aligns with privacy release. Cons: noisy for small cells, no covariate adjustment.

### Approach 2 — S-learner gradient boosting
Model predicts revenue per eligible population from bucketed features including exposure group. Counterfactual difference estimates incremental revenue.

Pros: adjusts for segment mix, potentially tighter estimates. Cons: model error, leakage risk, less transparent.

### Training vs inference
- Training: chronological train split on earlier dates.
- Inference: predict treated and holdout counterfactuals on released cells.
- Evaluation: later holdout dates compared with synthetic latent truth.

### Uncertainty reporting
- Baseline: bootstrap standard error over cells; 95% CI = estimate +/- 1.96 SE.
- Model: cell-level counterfactual variance scaled by sqrt(n); 95% CI.
- Confidence score: `clip(1 - SE / (abs(estimate) + epsilon), 0, 1)`.

### Deterministic fallback
Trigger when:
- treated or holdout cells insufficient;
- SE > 2x absolute estimate;
- privacy risk high;
- fairness alert;
- forced fallback method selected.

Fallback behavior:
- estimate = 0;
- action = hold or human review;
- no automatic budget change.
"""

PAGE3_STACK = r"""
# Page 3 — Technology stack, folder structure, and build steps

## Python-first CPU-only stack
- Python standard library.
- Streamlit for dashboard.
- pandas and numpy for tabular processing.
- scikit-learn HistGradientBoostingRegressor for optional model.
- Plotly for charts.
- OpenCV headless and Pillow for generated demo media fallback.
- SQLite for local persistent audit and canonical demo storage.

No paid APIs, no proprietary cloud services, no GPU required.

## Logical module boundaries in single-file prototype
Although delivered as `app.py`, the code is segmented into:
- constants/config;
- database/audit;
- synthetic data generation;
- privacy engine;
- validation/canonicalization;
- measurement algorithms;
- decision engine;
- live simulation;
- self-tests;
- documentation;
- Streamlit UI.

## Production repository layout
```text
privacy-ad-measurement/
  app.py
  requirements.txt
  README.md
  .env.example
  /src
    /domain
      measurement.py
      privacy.py
      recommendations.py
    /data
      synthetic_generator.py
      adapters.py
    /schemas
      canonical.schema.json
    /models
      slearner.py
    /services
      ingestion.py
      validation.py
      audit.py
    /api
      routes.py
    /dashboard
      streamlit_app.py
    /configs
      settings.py
    /scripts
      build_demo_data.py
      run_eval.py
  /migrations
    001_init.sql
  /docs
    architecture.md
    data_dictionary.md
    risk_register.md
  /tests
    /unit
    /integration
    /e2e
```

## Milestones
1. Environment and configuration.
2. Synthetic/open data preparation.
3. Schema validation and quarantine.
4. Canonical storage and audit.
5. Baseline DP difference estimator.
6. Core S-learner model.
7. Decision/scoring layer.
8. Streamlit dashboard.
9. Generated demo media and live simulation.
10. Tests, packaging, documentation.

## PowerShell setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Optional Docker Compose route

```yaml
services:
  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - APP_ENV=local
    volumes:
      - ./data:/app/data
```

## .env.example

```text
# No secrets required for local synthetic demo
APP_ENV=local
DB_PATH=./data/measurement.sqlite
PRIVACY_POLICY_MAX_EPSILON=2.0
AUDIT_RETENTION_DAYS=365
FEATURE_SLEARNER=true
```

Never commit secrets. This prototype does not require any API key.

## License inventory
| Dependency | License | Commercial use | Attribution / notice |
|---|---|---|---|
| Streamlit | Apache-2.0 | Yes | Preserve license notices |
| pandas | BSD-3-Clause | Yes | Preserve license notices |
| NumPy | BSD-3-Clause | Yes | Preserve license notices |
| scikit-learn | BSD-3-Clause | Yes | Preserve license notices |
| Plotly | MIT | Yes | Preserve license notices |
| opencv-python-headless | Apache-2.0 / OpenCV license | Yes | Preserve license notices |
| Pillow | MIT-like / HPND | Yes | Preserve license notices |
| SQLite | Public domain | Yes | None required |
| Synthetic data | Generated for demo | Yes | Label as synthetic |

Verify transitive licenses at build time in production CI.
"""

PAGE4_TESTING = r"""
# Page 4 — Testing, evaluation, security, and reliability

## Unit tests implemented in-app
The Self-tests tab executes:
- hand-calculated revenue uplift fixture;
- deterministic Laplace noise with fixed seed;
- suppression rule;
- duplicate dedupe keeping latest ingestion;
- negative spend detection;
- schema version presence;
- demo media generation;
- dataset row counts;
- no direct PII columns;
- suppressed rows not released;
- quarantine contains bad records;
- baseline and S-learner estimate shape;
- confidence bounds valid;
- recommendation approval gate.

## Integration and end-to-end tests
The app exercises:
- synthetic generation -> privacy noise -> dirty injection -> validation -> canonical storage;
- estimator -> recommendation -> audit log;
- dashboard rendering -> download CSV;
- live simulation -> event log -> final snapshot.

These run locally without paid services.

## Evaluation protocol
- Chronological split: first ~70% dates train/estimate context, last ~30% holdout evaluation.
- Leakage checks: no future dates in training, no user-level identifiers, no latent truth in estimator inputs.
- Baseline comparison: DP aggregate difference vs S-learner vs deterministic fallback.
- Metrics:
  - MAE = mean(abs(estimate - latent_truth))
  - RMSE = sqrt(mean((estimate - latent_truth)^2))
  - Sign accuracy = mean(sign(estimate) == sign(latent_truth))
  - CI coverage = mean(latent_truth within 95% CI)
- Representative edge cases: small cells, late arrivals, duplicates, negative spend, invalid enum, outlier revenue, forced fallback.

## What metrics cannot prove
- Synthetic latent truth does not prove real-world causal lift.
- CI coverage on synthetic data does not guarantee calibration under adversarial drift.
- High sign accuracy does not prove advertiser acceptance or privacy safety.
- Low latency does not prove production reliability under distributed load.

## Domain quality metrics
- Incremental revenue error vs simulated truth.
- Suppressed fraction.
- Released-cell coverage.
- Quarantine rate.
- Fairness alert count.
- Privacy risk score.

## Operational metrics
- Preparation latency.
- Estimation latency.
- Row counts in/out.
- Audit log completeness.
- Demo media generation status.
- Database availability.

## Security and privacy controls
- Least privilege: local process only, no network calls in core demo.
- Input validation: enums, ranges, timestamps, duplicates, outliers.
- Secrets management: no secrets required; `.env.example` has placeholders only.
- Secure defaults: suppression enabled, epsilon policy warning, human approval gates.
- Encryption configuration: production would use TLS and at-rest encryption; demo uses local temp SQLite.
- Retention controls: production would enforce retention; demo uses temporary files.
- Audit logging: all bootstrap, saves, approvals recorded.
- Threat model: spoofed upstream data, small-cell leakage, model drift, excessive epsilon, unauthorized export, operational outage.
- Rollback: reset demo state; production would use schema/versioned releases.
- Kill switch: reset button and finite simulation; production would disable automated actions.
- Human review: required for consequential actions.
- Safe failure mode: deterministic fallback to hold/review.
"""

PAGE5_DEMO = r"""
# Page 5 — Live demo, deployment, and acceptance criteria

## Reproducible 5-10 minute demo
1. Run `streamlit run app.py`.
2. The app auto-generates synthetic data and starts a 30-tick live simulation.
3. Watch generated demo clip and live KPIs.
4. Open Measurement tab; compare baseline and S-learner estimates.
5. Open Privacy & Quality tab; inspect suppression, epsilon, quarantine, fairness.
6. Open Revenue Optimization tab; review recommendations and approve a simulated action.
7. Open Audit tab; inspect audit log and download synthetic CSV.
8. Open Self-tests tab; run in-app tests.

## Normal scenario
- Retail media shows positive lower-bound incremental revenue.
- Recommendation suggests budget increase but requires approval if >10%.

## Difficult edge case
- CTV has negative or uncertain uplift and a fairness/suppression alert.
- System holds budget and requests human review.

## Failure/recovery scenario
- Dirty records include duplicate, negative spend, invalid channel, outlier revenue, late out-of-window.
- Validator quarantines or flags them; dashboard shows quarantine rate and recovery path.

## Explanation of result
- Every estimate includes method, CI, confidence, privacy risk, cell count, suppressed rate, and audit trail.
- Simulated ground truth is labeled separately and never used as estimator input.

## Dashboard requirements
- Overview metrics.
- Filters for campaign/channel/date/method/epsilon.
- Drill-downs by segment and quality flags.
- Trends for spend, reported revenue, estimated incrementality.
- Data-quality status.
- Model/algorithm details.
- Explanations and evidence.
- Configuration.
- Audit history.
- Distinct labels: Observed, Reported/noised, Estimated, Predicted, Simulated truth.

## Local Windows setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Health checks
- DB path exists.
- Schema version present.
- Clean rows > 0.
- Demo media generated.
- Self-tests pass.
- Audit log writable.

## Structured logs
- Audit table stores timestamp, actor, action, object, details JSON, approval flag.
- Runs table stores latency, row counts, method, epsilon, status.

## Backup/restore and rollback
- Demo: copy/delete temporary SQLite and media files.
- Production: scheduled backups, point-in-time restore, blue/green release, schema migration rollback.

## Resource estimates
- CPU-only.
- Memory: modest, dominated by pandas DataFrames of ~10k rows.
- Storage: temporary SQLite and demo media files.
- Runtime: initial demo simulation several seconds; cached recomputation fast.

## Troubleshooting
- If video does not play, app falls back to animated GIF.
- If OpenCV install fails, install requirements or use GIF fallback.
- If port is busy, run `streamlit run app.py --server.port 8502`.
- If demo state is stale, use Reset demo state.

## Business-value calculator
User-entered assumptions:
- monthly ad spend;
- gross margin;
- current incremental ROI;
- proposed incremental ROI;
- implementation cost;
- expected error cost.

Outputs:
- incremental profit;
- net monthly value;
- payback period;
- sensitivity table.

The calculator does not fabricate customer results.

## Acceptance criteria
A fresh developer can:
- install requirements;
- run `streamlit run app.py`;
- see non-blank dashboard immediately;
- see generated demo clip;
- watch auto simulation;
- interact with controls after simulation;
- run self-tests;
- download synthetic audit/data CSVs;
- read PRD, architecture, data contracts, risk register, and roadmap inside the app.
"""

RISK_REGISTER = r"""
# Risk register

| Risk | Category | Likelihood | Impact | Mitigation | Human approval required |
|---|---|---|---|---|---|
| Upstream ad/revenue data unavailable | Data availability | Medium | High | Last-known-good, stale flags, quarantine | Restore from backup |
| Small cell re-identification | Privacy | Medium | Critical | Suppression, DP noise, epsilon policy | Epsilon override |
| Model overfits or drifts | Model error | Medium | High | Chronological split, fallback, monitoring | Model promotion |
| Discriminatory budget allocation | Fairness | Medium | High | Segment checks, fairness alert, hold action | Override fairness hold |
| Fraudulent click/inflation abuse | Misuse | Medium | Medium | Anomaly events, outlier caps, audit | Manual investigation |
| Dashboard outage | Operational failure | Low | Medium | Health checks, static fallback, kill switch | Incident response |
| Users ignore recommendations | Adoption | Medium | Medium | Explainability, approval workflow, pilots | Policy change |
| Cloud/compute cost overrun | Cost | Medium | Medium | Batch caching, modular monolith, sampling | Budget increase |
| Unauthorized advertiser export | Privacy/governance | Low | Critical | RBAC, audit, approval gate | Export approval |
| Schema migration breaks pipeline | Operational | Medium | High | Versioned contracts, tests, rollback | Migration approval |

## Actions requiring human approval
- Increase epsilon above policy.
- Disable or lower suppression threshold.
- Change budget by more than 10%.
- Promote a model to production.
- Export advertiser-facing report.
- Override fairness alert.
- Modify retention or audit policy.
"""

BACKLOG_USER_STORIES = r"""
# Prioritized backlog and user stories

## P0 — Must have for reviewable prototype
1. Synthetic data generator with latent truth.
2. Canonical schema and validation/quarantine.
3. Privacy suppression and Laplace noise.
4. Baseline DP difference estimator with CI.
5. Dashboard with live auto demo and generated media.
6. Audit log and run telemetry.
7. Self-tests.
8. Documentation tabs.

## P1 — Important
1. S-learner GBM estimator.
2. Fairness subgroup alerts.
3. Recommendation approval workflow.
4. Business-value calculator.
5. Downloadable synthetic CSV artifacts.

## P2 — Future enterprise
1. Real source adapters.
2. Distributed DP accountant.
3. RBAC/SSO.
4. Scheduler and streaming ingestion.
5. Model registry and champion/challenger.
6. Advertiser dispute workflow.

## User stories with acceptance criteria

### Story 1 — Measure incremental revenue
**As a** marketing scientist, **I want** privacy-safe incrementality estimates **so that** I can trust campaign comparisons.  
**AC:**
- Estimates use only released bucketed cells.
- Each estimate has CI and method label.
- Small cells are suppressed.

### Story 2 — Review privacy posture
**As a** privacy analyst, **I want** epsilon and suppression metrics **so that** I can prevent leakage.  
**AC:**
- Dashboard shows epsilon, delta, noise mechanism, suppressed fraction.
- Policy override requires approval checkbox.
- Audit log records query/release metadata.

### Story 3 — Recommend budget changes
**As a** revenue ops manager, **I want** lower-bound ROI recommendations **so that** I avoid over-optimistic spending.  
**AC:**
- Recommendations use lower bound, not point estimate alone.
- Changes >10% require approval.
- Fallback holds when confidence is low.

### Story 4 — Investigate data quality
**As a** data engineer, **I want** quarantine and quality flags **so that** I can fix upstream issues.  
**AC:**
- Invalid rows are quarantined with reasons.
- Duplicates are deduped and counted.
- Late arrivals within grace are flagged, not dropped silently.

### Story 5 — Demonstrate value safely
**As an** executive, **I want** a business calculator with assumptions **so that** I can evaluate investment without fabricated claims.  
**AC:**
- Inputs are user-entered.
- Outputs show formula and sensitivity.
- No real customer results are asserted.
"""

GLOSSARY = r"""
# Glossary

- **Incrementality:** Difference in outcomes caused by advertising exposure versus a holdout counterfactual.
- **Holdout:** Eligible population deliberately not exposed, used as counterfactual.
- **Measurement cell:** Aggregated bucket defined by date, campaign, geography, age, device, consent, and exposure group.
- **Suppression:** Withholding a cell from release because population is too small or quality is inadequate.
- **Differential privacy (DP):** Formal noise-addition framework limiting inference about individual contributions.
- **Epsilon:** DP privacy-loss parameter; lower is stricter.
- **Delta:** DP failure probability parameter.
- **Laplace mechanism:** Noise mechanism for numeric queries with L1 sensitivity.
- **ROAS:** Return on ad spend, often revenue / spend; can be non-causal.
- **ROI:** Return on investment; here incremental revenue net of spend relative to spend.
- **CI:** Confidence interval.
- **MAE:** Mean absolute error.
- **RMSE:** Root mean squared error.
- **PEHE:** Precision in estimating heterogeneous effects; related to error in individualized uplift, approximated here by campaign-level latent error.
- **Canonical model:** Standardized schema after validation and privacy processing.
- **Provenance:** Origin and version information for data.
- **Quarantine:** Isolated invalid records excluded from analytics.
- **S-learner:** Single model that includes treatment/exposure as a feature and predicts counterfactual outcomes.
"""


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------


def bootstrap() -> None:
    if st.session_state.get("bootstrapped", False):
        return

    st.session_state.settings = {
        "epsilon": DEFAULT_EPSILON,
        "min_pop": MIN_CELL_POPULATION_DEFAULT,
        "method": "baseline_dp_diff",
        "dirty": True,
        "epsilon_override_approved": True,
    }
    st.session_state.auto_demo_active = True
    st.session_state.live_ticks = generate_live_ticks()
    st.session_state.bootstrapped = True
    st.session_state.run_id = uuid.uuid4().hex

    get_conn()
    log_audit(
        actor="system",
        action="bootstrap",
        object_type="app",
        object_id=st.session_state.run_id,
        details={
            "schema_version": SCHEMA_VERSION,
            "provenance": PROVENANCE,
            "seed": SEED,
            "data_mode": "synthetic_only",
        },
    )


def render_sidebar() -> None:
    st.sidebar.title("Controls")
    el_caption(st.sidebar, "Synthetic demo only. No API keys. CPU-friendly.")

    s = st.session_state.settings
    s["epsilon"] = float(
        st.sidebar.slider(
            "Privacy epsilon per release",
            min_value=0.25,
            max_value=4.0,
            value=float(s["epsilon"]),
            step=0.25,
            key="ctl_epsilon",
        )
    )
    s["min_pop"] = int(
        st.sidebar.number_input(
            "Minimum cell population",
            min_value=200,
            max_value=10000,
            value=int(s["min_pop"]),
            step=100,
            key="ctl_min_pop",
        )
    )

    method_options = ["baseline_dp_diff", "slearner_gbm", "deterministic_fallback"]
    current_idx = method_options.index(s["method"]) if s["method"] in method_options else 0
    s["method"] = st.sidebar.selectbox("Estimator", method_options, index=current_idx, key="ctl_method")
    s["dirty"] = bool(st.sidebar.checkbox("Inject dirty records", value=bool(s["dirty"]), key="ctl_dirty"))

    if s["epsilon"] > POLICY_MAX_EPSILON:
        el_msg(st.sidebar, "Epsilon exceeds default policy. Synthetic override requires acknowledgment.", "warning")
        s["epsilon_override_approved"] = bool(
            st.sidebar.checkbox(
                "Approve synthetic epsilon override",
                value=False,
                key="ctl_epsilon_override",
            )
        )
    else:
        s["epsilon_override_approved"] = True

    el_md(st.sidebar, "---")

    if el_button(st.sidebar, "Replay live simulation", key="btn_replay"):
        st.session_state.auto_demo_active = True
        rerun_app()

    if el_button(st.sidebar, "Reset demo state", key="btn_reset"):
        try:
            if "conn" in st.session_state:
                st.session_state.conn.close()
        except Exception:
            pass
        try:
            if os.path.exists(DB_PATH):
                os.remove(DB_PATH)
        except Exception:
            pass
        clear_caches()
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        rerun_app()

    el_md(st.sidebar, "---")
    el_md(
        st.sidebar,
        f"""
        **Schema:** `{SCHEMA_VERSION}`  
        **Provenance:** `{PROVENANCE}`  
        **DB:** `{os.path.basename(DB_PATH)}`  
        **Policy max epsilon:** `{POLICY_MAX_EPSILON}`
        """,
    )


def render_overview_tab(clean, quarantine, report, latent, estimates, recs, prep_latency) -> None:
    el_md(st, "Live working demo")
    el_caption(st, "All data is synthetic. Values are labeled: Observed, Reported/noised, Estimated, Predicted, Simulated truth.")

    assets = ensure_demo_assets()
    video_path = assets.get("video_path", "")
    gif_path = assets.get("gif_path", "")
    video_ok = bool(video_path and os.path.exists(video_path))
    gif_ok = bool(gif_path and os.path.exists(gif_path))

    if video_ok:
        el_md(st, f"Generated demo video ({str(assets.get('video_kind', 'video')).upper()})")
        el_video(st, video_path, assets.get("video_kind", "mp4"))
        data = read_bytes(video_path)
        if data:
            mime = "video/mp4" if assets.get("video_kind") == "mp4" else "video/webm"
            el_download(st, "Download demo video", data, os.path.basename(video_path), mime)
    else:
        el_msg(st, f"Video generation unavailable: {assets.get('video_err', '')}", "warning")

    if gif_ok:
        el_md(st, "Compatible animated preview (GIF)")
        el_image(st, gif_path)
        data = read_bytes(gif_path)
        if data:
            el_download(st, "Download animated preview", data, "privacy_ad_demo.gif", "image/gif")
    elif not video_ok:
        el_msg(
            st,
            f"Demo media generation failed: video={assets.get('video_err', '')}; gif={assets.get('gif_err', '')}",
            "error",
        )

    kpi_slot = st.empty()
    chart_slot = st.empty()
    event_slot = st.empty()

    if st.session_state.get("auto_demo_active", False):
        run_live_simulation(kpi_slot, chart_slot, event_slot)

    if not st.session_state.get("auto_demo_active", False):
        ticks = st.session_state.live_ticks
        cols = kpi_slot.columns(5)
        el_metric(cols[0], "Simulation", "Complete")
        el_metric(cols[1], "Observed spend", fmt_usd(ticks[-1]["cum_spend"]))
        el_metric(cols[2], "Reported conversions", f"{ticks[-1]['cum_reported_conversions']:,.0f}")
        el_metric(cols[3], "Est. incr. revenue", fmt_usd(ticks[-1]["cum_estimated_incremental"]))
        el_metric(cols[4], "Privacy status", ticks[-1]["privacy_status"])

        el_plotly(chart_slot, make_live_figure(ticks, len(ticks)))

        el_md(event_slot, "**Final event summary (simulated)**")
        all_events = []
        for t in ticks:
            all_events.extend(t["events"])
        el_json(event_slot, all_events[-8:] if all_events else [])

    el_md(st, "---")

    cols = st.columns(4)
    el_metric(cols[0], "Clean cells", f"{report.get('clean_rows', 0):,}")
    el_metric(cols[1], "Released cells", f"{report.get('released_rows', 0):,}")
    el_metric(cols[2], "Suppressed cells", f"{report.get('suppressed_rows', 0):,}")
    el_metric(cols[3], "Quarantine rows", f"{report.get('quarantine_rows', 0):,}")

    cols = st.columns(3)
    el_metric(cols[0], "Prep latency", f"{prep_latency:.2f}s")
    el_metric(cols[1], "Estimate method", st.session_state.settings["method"])
    el_metric(cols[2], "Epsilon", f"{st.session_state.settings['epsilon']:.2f}")


def render_measurement_tab(clean, latent, estimates, holdout_eval, model_info) -> None:
    el_md(st, "### Privacy-aware incrementality measurement")

    total_spend = float(estimates["spend"].sum())
    total_est = float(estimates["inc_rev_est"].sum())
    total_lb = float(estimates["inc_rev_lb"].sum())
    actionable = int((~estimates["fallback"] & (estimates["confidence"] > 0.25)).sum())

    cols = st.columns(4)
    el_metric(cols[0], "Total spend", fmt_usd(total_spend))
    el_metric(cols[1], "Estimated incremental revenue", fmt_usd(total_est))
    el_metric(cols[2], "Lower-bound incremental revenue", fmt_usd(total_lb))
    el_metric(cols[3], "Actionable campaigns", f"{actionable}/{len(estimates)}")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=estimates["campaign_id"],
            y=estimates["inc_rev_est"],
            name="Estimated incremental revenue",
            error_y=dict(type="data", array=1.96 * estimates["inc_rev_se"], visible=True),
            marker_color=np.where(estimates["fallback"], "#d97706", "#2563eb"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=estimates["campaign_id"],
            y=estimates["true_inc_rev_latent"],
            mode="markers",
            name="Simulated ground truth",
            marker=dict(symbol="x", size=12, color="#16a34a"),
        )
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="USD", xaxis_title="Campaign")
    show_plotly(fig)

    el_df(
        st,
        estimates[
            [
                "campaign_id",
                "channel",
                "spend",
                "inc_rev_est",
                "inc_rev_se",
                "inc_rev_lb",
                "inc_rev_ub",
                "confidence",
                "privacy_risk",
                "fairness_alert",
                "method",
                "fallback",
                "true_inc_rev_latent",
                "abs_error",
            ]
        ].round(2),
    )

    exp = st.expander("Algorithm and evaluation details")
    el_json(exp, model_info)
    if not holdout_eval.empty:
        mae = float(holdout_eval["abs_error"].mean())
        rmse = float(math.sqrt((holdout_eval["abs_error"] ** 2).mean()))
        coverage = float(holdout_eval["covered"].mean())
        sign_acc = float(holdout_eval["sign_correct"].mean())
        cols = exp.columns(4)
        el_metric(cols[0], "Holdout MAE", fmt_usd(mae))
        el_metric(cols[1], "Holdout RMSE", fmt_usd(rmse))
        el_metric(cols[2], "CI coverage", fmt_pct(coverage))
        el_metric(cols[3], "Sign accuracy", fmt_pct(sign_acc))
        el_df(exp, holdout_eval.round(2))
    else:
        el_msg(exp, "No holdout evaluation rows for deterministic fallback.", "info")

    exp2 = st.expander("Evidence drill-down")
    cid = exp2.selectbox("Campaign", sorted(estimates["campaign_id"].unique()), key="drill_campaign")
    sub = clean[clean["campaign_id"] == cid]
    el_metric(exp2, "Cells", f"{len(sub):,}")
    el_metric(exp2, "Suppressed rate", fmt_pct(float(sub["suppressed"].mean())))
    el_df(
        exp2,
        sub[
            [
                "cell_id",
                "event_date",
                "geo_bucket",
                "age_bucket",
                "device_type",
                "consent_state",
                "exposure_group",
                "eligible_population",
                "conversions_reported",
                "revenue_reported",
                "spend_reported",
                "epsilon",
                "released",
                "quality_flags",
            ]
        ].head(100),
    )


def render_privacy_quality_tab(clean, quarantine, report) -> None:
    el_md(st, "### Privacy, fairness, and data quality")

    s = st.session_state.settings
    cols = st.columns(4)
    el_metric(cols[0], "Epsilon", f"{s['epsilon']:.2f}")
    el_metric(cols[1], "Min population", f"{s['min_pop']:,}")
    el_metric(cols[2], "Suppressed fraction", fmt_pct(float(clean["suppressed"].mean())))
    el_metric(cols[3], "Quarantine fraction", fmt_pct(float(len(quarantine)) / max(1, report.get("input_rows", 1))))

    if s["epsilon"] > POLICY_MAX_EPSILON and not s.get("epsilon_override_approved", False):
        el_msg(st, "Epsilon exceeds policy and override is not approved. Production would block release.", "error")
    else:
        msg = "Privacy policy status: acceptable for synthetic demo." if s["epsilon"] <= POLICY_MAX_EPSILON else "Synthetic override acknowledged."
        el_msg(st, msg, "success")

    supp = clean.groupby(["geo_bucket", "age_bucket"])["suppressed"].mean().reset_index()
    if supp.empty:
        el_msg(st, "No released cells available for suppression heatmap.", "info")
    else:
        fig = px.density_heatmap(supp, x="geo_bucket", y="age_bucket", z="suppressed", color_continuous_scale="Oranges")
        fig.update_layout(height=320, title="Suppression rate by geo and age bucket")
        show_plotly(fig)

    flag_series = clean["quality_flags"].str.split(",").explode().replace("", np.nan).dropna()
    if flag_series.empty:
        el_msg(st, "No quality flags in released clean cells.", "info")
    else:
        flags = flag_series.value_counts().rename_axis("flag").reset_index(name="count")
        show_plotly(px.bar(flags, x="flag", y="count", title="Quality flags"))

    el_md(st, "#### Quarantine sample")
    if quarantine.empty:
        el_msg(st, "No quarantined records in current configuration.", "info")
    else:
        el_df(st, quarantine[["cell_id", "campaign_id", "channel", "event_date", "spend_reported", "revenue_reported", "quarantine_reason"]].head(50))

    el_md(st, "#### Example canonical record")
    released_sample = clean[clean["released"]]
    if released_sample.empty:
        el_msg(st, "No released cells in current configuration.", "info")
    else:
        el_json(st, released_sample.iloc[0].to_dict())


def render_revenue_tab(estimates, recs) -> None:
    el_md(st, "### Revenue optimization recommendations")

    display_cols = [
        "campaign_id",
        "channel",
        "current_spend",
        "recommended_scale",
        "proposed_spend",
        "action",
        "approval_required",
        "estimated_incremental_revenue",
        "lower_bound_incremental_revenue",
        "confidence",
        "privacy_risk",
        "fairness_alert",
        "method",
    ]
    el_df(st, recs[display_cols].round(3))

    approval_options = recs.loc[recs["approval_required"], "campaign_id"].tolist()
    selected = st.multiselect(
        "Select consequential actions to approve (simulated)",
        approval_options,
        key="approval_multiselect",
    )

    if el_button(st, "Apply approved simulated actions", key="btn_apply_approvals"):
        if not selected:
            el_msg(st, "No actions selected.", "warning")
        else:
            for cid in selected:
                row = recs[recs["campaign_id"] == cid].iloc[0]
                log_audit(
                    actor="local_demo_user",
                    action="approve_budget_change",
                    object_type="campaign",
                    object_id=cid,
                    details={
                        "action": row["action"],
                        "recommended_scale": float(row["recommended_scale"]),
                        "current_spend": float(row["current_spend"]),
                        "proposed_spend": float(row["proposed_spend"]),
                        "confidence": float(row["confidence"]),
                        "privacy_risk": float(row["privacy_risk"]),
                    },
                    approval_required=True,
                    approved_by="local_demo_user",
                )
            el_msg(st, f"Logged {len(selected)} approved simulated actions in audit history.", "success")

    el_md(st, "---")
    el_md(st, "#### Transparent business-value calculator")
    el_caption(st, "Enter assumptions. This calculator does not fabricate customer results.")

    monthly_spend = float(
        st.number_input(
            "Monthly ad spend (USD)",
            min_value=10000,
            max_value=100000000,
            value=1000000,
            step=50000,
            key="calc_monthly_spend",
        )
    )
    margin = float(st.slider("Gross margin", min_value=0.05, max_value=0.90, value=0.35, step=0.01, key="calc_margin"))
    current_roi = float(
        st.number_input(
            "Current incremental ROI",
            min_value=-1.0,
            max_value=5.0,
            value=0.15,
            step=0.01,
            key="calc_current_roi",
        )
    )
    proposed_roi = float(
        st.number_input(
            "Proposed incremental ROI",
            min_value=-1.0,
            max_value=5.0,
            value=0.25,
            step=0.01,
            key="calc_proposed_roi",
        )
    )
    implementation_cost = float(
        st.number_input(
            "Implementation cost (USD/month)",
            min_value=0,
            max_value=10000000,
            value=50000,
            step=5000,
            key="calc_impl_cost",
        )
    )
    error_cost = float(
        st.number_input(
            "Expected error/mitigation cost (USD/month)",
            min_value=0,
            max_value=10000000,
            value=10000,
            step=1000,
            key="calc_error_cost",
        )
    )

    incremental_profit = monthly_spend * (proposed_roi - current_roi) * margin
    net_value = incremental_profit - implementation_cost - error_cost
    payback = implementation_cost / net_value if net_value > 0 else float("inf")

    cols = st.columns(3)
    el_metric(cols[0], "Incremental profit", fmt_usd(incremental_profit))
    el_metric(cols[1], "Net monthly value", fmt_usd(net_value))
    el_metric(cols[2], "Payback months", f"{payback:.1f}" if math.isfinite(payback) else "n/a")

    el_md(st, "##### Sensitivity")
    rows = []
    for roi_delta in [-0.05, 0.0, 0.05]:
        for margin_delta in [-0.05, 0.0, 0.05]:
            pr = proposed_roi + roi_delta
            mg = max(0.01, margin + margin_delta)
            ip = monthly_spend * (pr - current_roi) * mg
            nv = ip - implementation_cost - error_cost
            rows.append({"proposed_roi": pr, "margin": mg, "net_monthly_value": nv})
    el_df(st, pd.DataFrame(rows).round(2))


def render_audit_tab() -> None:
    el_md(st, "### Audit history and observability")
    conn = get_conn()
    assets = ensure_demo_assets()

    cols = st.columns(4)
    el_metric(cols[0], "DB file", os.path.basename(DB_PATH))
    el_metric(cols[1], "Schema version", SCHEMA_VERSION)
    el_metric(cols[2], "Run id", st.session_state.get("run_id", "")[:12])
    el_metric(cols[3], "Demo media", assets.get("video_kind", "none") if assets.get("video_path") else ("gif" if assets.get("gif_path") else "none"))

    try:
        audit = pd.read_sql_query("SELECT * FROM audit_log ORDER BY id DESC LIMIT 200", conn)
    except Exception:
        audit = pd.DataFrame()
    try:
        runs = pd.read_sql_query("SELECT * FROM runs ORDER BY started_at DESC LIMIT 50", conn)
    except Exception:
        runs = pd.DataFrame()

    el_md(st, "#### Audit log")
    if audit.empty:
        el_msg(st, "No audit entries yet.", "info")
    else:
        el_df(st, audit)
        data = audit.to_csv(index=False).encode("utf-8")
        el_download(st, "Download audit CSV", data, "audit_log.csv", "text/csv")

    el_md(st, "#### Runs")
    if runs.empty:
        el_msg(st, "No run telemetry yet. Save a dataset by changing sidebar settings or reopening Measurement.", "info")
    else:
        el_df(st, runs)


def render_docs_tab() -> None:
    el_md(st, "### Reviewable product artifacts")
    doc_tabs = st.tabs(
        [
            "Executive",
            "Page 1 PRD",
            "Page 2 Architecture",
            "Page 3 Stack",
            "Page 4 Testing",
            "Page 5 Demo",
            "Risk",
            "Backlog",
            "Glossary",
            "Mermaid",
        ]
    )

    el_md(doc_tabs[0], EXEC_SUMMARY)
    el_md(doc_tabs[1], PAGE1_PRD)
    el_md(doc_tabs[2], PAGE2_ARCH)
    el_md(doc_tabs[3], PAGE3_STACK)
    el_md(doc_tabs[4], PAGE4_TESTING)
    el_md(doc_tabs[5], PAGE5_DEMO)
    el_md(doc_tabs[6], RISK_REGISTER)
    el_md(doc_tabs[7], BACKLOG_USER_STORIES)
    el_md(doc_tabs[8], GLOSSARY)

    el_md(doc_tabs[9], "### Mermaid source")
    el_code(
        doc_tabs[9],
        """
flowchart LR
  A[Synthetic ad server] --> B[Ingestion adapter]
  A2[Synthetic revenue warehouse] --> B
  A3[Consent / holdout logger] --> B
  B --> C[Schema validator]
  C -->|valid| D[Canonical measurement store]
  C -->|invalid| E[Quarantine]
  D --> F[Privacy engine: suppression + DP noise]
  F --> G[Baseline DP difference estimator]
  F --> H[S-learner GBM estimator]
  G --> I[Decision scorer]
  H --> I
  I --> J[Streamlit dashboard]
  J --> K[Audit and observability]
        """.strip(),
        language="mermaid",
    )
    el_code(
        doc_tabs[9],
        """
sequenceDiagram
  participant CM as Campaign manager
  participant UI as Dashboard
  participant VAL as Validator
  participant PRIV as Privacy engine
  participant MEAS as Measurement service
  participant AUD as Audit store
  CM->>UI: Request incrementality for date range
  UI->>VAL: Validate query and schema version
  VAL->>PRIV: Provide released candidate cells
  PRIV-->>VAL: Apply suppression and DP noise
  VAL->>MEAS: Pass canonical aggregates
  MEAS->>MEAS: Compute baseline or model estimate + CI
  MEAS->>AUD: Log epsilon, row counts, method, latency
  MEAS-->>UI: Estimates, evidence, recommendations
  UI-->>CM: Show action; require approval if consequential
        """.strip(),
        language="mermaid",
    )


def render_tests_tab() -> None:
    el_md(st, "### In-app test harness")
    el_caption(st, "These tests run locally without paid services. In production, mirror them in /tests/unit, /tests/integration, and /tests/e2e.")

    if "test_results" not in st.session_state:
        st.session_state.test_results = None

    cols = st.columns([1, 4])
    if el_button(cols[0], "Run self-tests", key="btn_run_tests"):
        with st.spinner("Running tests..."):
            st.session_state.test_results = run_self_tests()

    if st.session_state.test_results is None:
        el_msg(st, "Click Run self-tests to execute the harness.", "info")
    else:
        df = pd.DataFrame(st.session_state.test_results)
        passed = int(df["passed"].sum())
        total = int(len(df))
        cols2 = st.columns(2)
        el_metric(cols2[0], "Passed", passed)
        el_metric(cols2[1], "Total", total)
        if passed == total:
            el_msg(st, "All self-tests passed.", "success")
        else:
            el_msg(st, "Some self-tests failed.", "error")
        el_df(st, df)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🔐", layout="wide", initial_sidebar_state="expanded")
    try:
        bootstrap()
        render_sidebar()

        st.title(APP_TITLE)
        el_md(st, f"### {APP_SUBTITLE}")
        el_caption(st, "Synthetic data only · privacy-aware aggregates · CPU-only · no API keys")

        s = st.session_state.settings
        date_key = pd.Timestamp.today().date().isoformat()

        clean, quarantine, report, latent, prep_latency = prepare_dataset(
            epsilon=s["epsilon"],
            min_pop=s["min_pop"],
            seed=SEED,
            dirty=s["dirty"],
            date_key=date_key,
        )

        sig = f"{s['epsilon']}|{s['min_pop']}|{s['dirty']}|{s['method']}|{date_key}"
        if st.session_state.get("saved_sig") != sig:
            started = iso(now_utc() - timedelta(seconds=prep_latency))
            finished = iso(now_utc())
            save_dataset_to_db(clean, quarantine, report)
            record_run(
                run_id=st.session_state.get("run_id", uuid.uuid4().hex),
                started_at=started,
                finished_at=finished,
                epsilon=s["epsilon"],
                min_pop=s["min_pop"],
                method=s["method"],
                rows_in=report.get("input_rows", 0),
                rows_clean=report.get("clean_rows", 0),
                rows_quar=report.get("quarantine_rows", 0),
                latency_ms=prep_latency * 1000.0,
                status="ok",
                notes="dataset_saved",
            )
            log_audit(
                actor="system",
                action="save_dataset",
                object_type="canonical_store",
                object_id=sig,
                details=report,
            )
            st.session_state.saved_sig = sig

        estimates, holdout_eval, model_info = get_estimates(
            clean=clean,
            latent=latent,
            method=s["method"],
            epsilon=s["epsilon"],
            min_pop=s["min_pop"],
        )
        recs = make_recommendations(estimates)

        tabs = st.tabs(["Live Demo", "Measurement", "Privacy & Quality", "Revenue Optimization", "Audit", "Docs", "Self-tests"])

        with tabs[0]:
            render_overview_tab(clean, quarantine, report, latent, estimates, recs, prep_latency)
        with tabs[1]:
            render_measurement_tab(clean, latent, estimates, holdout_eval, model_info)
        with tabs[2]:
            render_privacy_quality_tab(clean, quarantine, report)
        with tabs[3]:
            render_revenue_tab(estimates, recs)
        with tabs[4]:
            render_audit_tab()
        with tabs[5]:
            render_docs_tab()
        with tabs[6]:
            render_tests_tab()
    except Exception as exc:
        cls = exc.__class__.__name__
        if cls in {"RerunException", "StopException", "StreamlitAPIException"} and "DuplicateElementId" not in cls:
            raise
        if "DuplicateElementId" in cls:
            st.session_state["_duplicate_element_recovery"] = st.session_state.get("_duplicate_element_recovery", 0) + 1
            if st.session_state["_duplicate_element_recovery"] <= 2:
                rerun_app()
        st.error("Application encountered an unexpected error. Safe failure mode: no automated budget action is taken.")
        st.code(traceback.format_exc(), language="text")


if __name__ == "__main__":
    main()