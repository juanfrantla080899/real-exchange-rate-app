"""
Real Exchange Rate (RER) Explorer
International Finance class project.

Pulls monthly CPI and exchange-rate data from the IMF SDMX 3.0 API and plots:
  * Nominal S(t): units of Country 2 currency per 1 unit of Country 1 currency
  * RER(t) = S(t) * P1(t) / P2(t)
  * Real Index(t) = 100 * RER(t) / RER*, where RER* is the average RER over
    the user-selected sample (valid months only)
  * PPP-implied exchange rate S_PPP(t) = RER* * P2(t) / P1(t)

Run with:  streamlit run app.py
"""

from __future__ import annotations

import io
import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMF_BASE_URL = os.environ.get("IMF_SDMX_BASE", "https://api.imf.org/external/sdmx/3.0")
IMF_AGENCY = os.environ.get("IMF_AGENCY", "IMF.STA")
CPI_DATAFLOW = os.environ.get("IMF_CPI_DATAFLOW", "CPI")
ER_DATAFLOW = os.environ.get("IMF_ER_DATAFLOW", "ER")

CPI_KEY = "{iso3}.CPI._T.IX.M"          # monthly CPI index
ER_KEY = "{iso3}.XDC_USD.PA_RT.M"       # monthly average, local currency per USD

FIRST_MONTH = pd.Period("1976-01", freq="M")
CACHE_TTL_SECONDS = 24 * 60 * 60
REQUEST_TIMEOUT = 45

# A month-over-month move in ln S(t) larger than this is flagged as a possible
# currency redenomination or series break (ln(1.5) is about 0.405).
BREAK_LOG_THRESHOLD = float(np.log(1.5))

# Minimum observations to estimate the AR(1) half-life.
MIN_OBS_HALF_LIFE = 24

OTHER_LABEL = "Other (type an IMF ISO3 code)"

# ISO3 -> (name, currency). The IMF uses G163 for the Euro area.
COUNTRIES: dict[str, tuple[str, str]] = {
    "G163": ("Euro area", "EUR"),
    "AFG": ("Afghanistan", "AFN"), "ALB": ("Albania", "ALL"), "DZA": ("Algeria", "DZD"),
    "AGO": ("Angola", "AOA"), "ATG": ("Antigua and Barbuda", "XCD"), "ARG": ("Argentina", "ARS"),
    "ARM": ("Armenia", "AMD"), "ABW": ("Aruba", "AWG"), "AUS": ("Australia", "AUD"),
    "AUT": ("Austria", "EUR"), "AZE": ("Azerbaijan", "AZN"), "BHS": ("Bahamas", "BSD"),
    "BHR": ("Bahrain", "BHD"), "BGD": ("Bangladesh", "BDT"), "BRB": ("Barbados", "BBD"),
    "BLR": ("Belarus", "BYN"), "BEL": ("Belgium", "EUR"), "BLZ": ("Belize", "BZD"),
    "BEN": ("Benin", "XOF"), "BTN": ("Bhutan", "BTN"), "BOL": ("Bolivia", "BOB"),
    "BIH": ("Bosnia and Herzegovina", "BAM"), "BWA": ("Botswana", "BWP"), "BRA": ("Brazil", "BRL"),
    "BRN": ("Brunei Darussalam", "BND"), "BGR": ("Bulgaria", "EUR"), "BFA": ("Burkina Faso", "XOF"),
    "BDI": ("Burundi", "BIF"), "CPV": ("Cabo Verde", "CVE"), "KHM": ("Cambodia", "KHR"),
    "CMR": ("Cameroon", "XAF"), "CAN": ("Canada", "CAD"), "CAF": ("Central African Republic", "XAF"),
    "TCD": ("Chad", "XAF"), "CHL": ("Chile", "CLP"), "CHN": ("China", "CNY"),
    "COL": ("Colombia", "COP"), "COM": ("Comoros", "KMF"), "COD": ("Congo, Dem. Rep.", "CDF"),
    "COG": ("Congo, Rep.", "XAF"), "CRI": ("Costa Rica", "CRC"), "CIV": ("Cote d'Ivoire", "XOF"),
    "HRV": ("Croatia", "EUR"), "CYP": ("Cyprus", "EUR"), "CZE": ("Czech Republic", "CZK"),
    "DNK": ("Denmark", "DKK"), "DJI": ("Djibouti", "DJF"), "DMA": ("Dominica", "XCD"),
    "DOM": ("Dominican Republic", "DOP"), "ECU": ("Ecuador", "USD"), "EGY": ("Egypt", "EGP"),
    "SLV": ("El Salvador", "USD"), "GNQ": ("Equatorial Guinea", "XAF"), "ERI": ("Eritrea", "ERN"),
    "EST": ("Estonia", "EUR"), "SWZ": ("Eswatini", "SZL"), "ETH": ("Ethiopia", "ETB"),
    "FJI": ("Fiji", "FJD"), "FIN": ("Finland", "EUR"), "FRA": ("France", "EUR"),
    "GAB": ("Gabon", "XAF"), "GMB": ("Gambia", "GMD"), "GEO": ("Georgia", "GEL"),
    "DEU": ("Germany", "EUR"), "GHA": ("Ghana", "GHS"), "GRC": ("Greece", "EUR"),
    "GRD": ("Grenada", "XCD"), "GTM": ("Guatemala", "GTQ"), "GIN": ("Guinea", "GNF"),
    "GNB": ("Guinea-Bissau", "XOF"), "GUY": ("Guyana", "GYD"), "HTI": ("Haiti", "HTG"),
    "HND": ("Honduras", "HNL"), "HKG": ("Hong Kong SAR", "HKD"), "HUN": ("Hungary", "HUF"),
    "ISL": ("Iceland", "ISK"), "IND": ("India", "INR"), "IDN": ("Indonesia", "IDR"),
    "IRN": ("Iran", "IRR"), "IRQ": ("Iraq", "IQD"), "IRL": ("Ireland", "EUR"),
    "ISR": ("Israel", "ILS"), "ITA": ("Italy", "EUR"), "JAM": ("Jamaica", "JMD"),
    "JPN": ("Japan", "JPY"), "JOR": ("Jordan", "JOD"), "KAZ": ("Kazakhstan", "KZT"),
    "KEN": ("Kenya", "KES"), "KOR": ("Korea", "KRW"), "KWT": ("Kuwait", "KWD"),
    "KGZ": ("Kyrgyz Republic", "KGS"), "LAO": ("Lao P.D.R.", "LAK"), "LVA": ("Latvia", "EUR"),
    "LBN": ("Lebanon", "LBP"), "LSO": ("Lesotho", "LSL"), "LBR": ("Liberia", "LRD"),
    "LBY": ("Libya", "LYD"), "LTU": ("Lithuania", "EUR"), "LUX": ("Luxembourg", "EUR"),
    "MAC": ("Macao SAR", "MOP"), "MDG": ("Madagascar", "MGA"), "MWI": ("Malawi", "MWK"),
    "MYS": ("Malaysia", "MYR"), "MDV": ("Maldives", "MVR"), "MLI": ("Mali", "XOF"),
    "MLT": ("Malta", "EUR"), "MRT": ("Mauritania", "MRU"), "MUS": ("Mauritius", "MUR"),
    "MEX": ("Mexico", "MXN"), "MDA": ("Moldova", "MDL"), "MNG": ("Mongolia", "MNT"),
    "MNE": ("Montenegro", "EUR"), "MAR": ("Morocco", "MAD"), "MOZ": ("Mozambique", "MZN"),
    "MMR": ("Myanmar", "MMK"), "NAM": ("Namibia", "NAD"), "NPL": ("Nepal", "NPR"),
    "NLD": ("Netherlands", "EUR"), "NZL": ("New Zealand", "NZD"), "NIC": ("Nicaragua", "NIO"),
    "NER": ("Niger", "XOF"), "NGA": ("Nigeria", "NGN"), "MKD": ("North Macedonia", "MKD"),
    "NOR": ("Norway", "NOK"), "OMN": ("Oman", "OMR"), "PAK": ("Pakistan", "PKR"),
    "PAN": ("Panama", "PAB"), "PNG": ("Papua New Guinea", "PGK"), "PRY": ("Paraguay", "PYG"),
    "PER": ("Peru", "PEN"), "PHL": ("Philippines", "PHP"), "POL": ("Poland", "PLN"),
    "PRT": ("Portugal", "EUR"), "QAT": ("Qatar", "QAR"), "ROU": ("Romania", "RON"),
    "RUS": ("Russia", "RUB"), "RWA": ("Rwanda", "RWF"), "WSM": ("Samoa", "WST"),
    "SAU": ("Saudi Arabia", "SAR"), "SEN": ("Senegal", "XOF"), "SRB": ("Serbia", "RSD"),
    "SYC": ("Seychelles", "SCR"), "SLE": ("Sierra Leone", "SLE"), "SGP": ("Singapore", "SGD"),
    "SVK": ("Slovak Republic", "EUR"), "SVN": ("Slovenia", "EUR"), "SLB": ("Solomon Islands", "SBD"),
    "SOM": ("Somalia", "SOS"), "ZAF": ("South Africa", "ZAR"), "SSD": ("South Sudan", "SSP"),
    "ESP": ("Spain", "EUR"), "LKA": ("Sri Lanka", "LKR"), "KNA": ("St. Kitts and Nevis", "XCD"),
    "LCA": ("St. Lucia", "XCD"), "VCT": ("St. Vincent and the Grenadines", "XCD"),
    "SDN": ("Sudan", "SDG"), "SUR": ("Suriname", "SRD"), "SWE": ("Sweden", "SEK"),
    "CHE": ("Switzerland", "CHF"), "SYR": ("Syria", "SYP"), "TJK": ("Tajikistan", "TJS"),
    "TZA": ("Tanzania", "TZS"), "THA": ("Thailand", "THB"), "TLS": ("Timor-Leste", "USD"),
    "TGO": ("Togo", "XOF"), "TON": ("Tonga", "TOP"), "TTO": ("Trinidad and Tobago", "TTD"),
    "TUN": ("Tunisia", "TND"), "TUR": ("Turkiye", "TRY"), "TKM": ("Turkmenistan", "TMT"),
    "UGA": ("Uganda", "UGX"), "UKR": ("Ukraine", "UAH"), "ARE": ("United Arab Emirates", "AED"),
    "GBR": ("United Kingdom", "GBP"), "USA": ("United States", "USD"), "URY": ("Uruguay", "UYU"),
    "UZB": ("Uzbekistan", "UZS"), "VUT": ("Vanuatu", "VUV"), "VEN": ("Venezuela", "VES"),
    "VNM": ("Vietnam", "VND"), "YEM": ("Yemen", "YER"), "ZMB": ("Zambia", "ZMW"),
    "ZWE": ("Zimbabwe", "ZWG"),
}

# Euro adoption month for current members: before this month the IMF series
# may be in the legacy currency (DEM, FRF, ...), so S(t) can show a level break.
EURO_ADOPTION = {
    "AUT": "1999-01", "BEL": "1999-01", "FIN": "1999-01", "FRA": "1999-01", "DEU": "1999-01",
    "IRL": "1999-01", "ITA": "1999-01", "LUX": "1999-01", "NLD": "1999-01", "PRT": "1999-01",
    "ESP": "1999-01", "GRC": "2001-01", "SVN": "2007-01", "CYP": "2008-01", "MLT": "2008-01",
    "SVK": "2009-01", "EST": "2011-01", "LVA": "2014-01", "LTU": "2015-01", "HRV": "2023-01",
    "BGR": "2026-01", "G163": "1999-01",
}

PRESETS = {
    "Choose a preset...": None,
    "Mexico vs United States (MXN/USD)": ("MEX", "USA"),
    "Euro area vs United States (EUR/USD)": ("G163", "USA"),
    "Japan vs United States (JPY/USD)": ("JPN", "USA"),
    "United Kingdom vs United States (GBP/USD)": ("GBR", "USA"),
    "China vs United States (CNY/USD)": ("CHN", "USA"),
    "Brazil vs Mexico (BRL/MXN)": ("BRA", "MEX"),
    "Switzerland vs Euro area (CHF/EUR)": ("CHE", "G163"),
}

COLOR_INDEX = "#1f5aa6"
COLOR_NOMINAL = "#c2620a"
COLOR_PPP = "#5b6470"
FILL_OVER = "rgba(200, 55, 55, 0.16)"
FILL_UNDER = "rgba(35, 140, 90, 0.16)"


def label_for(code: str) -> str:
    name, cur = COUNTRIES[code]
    return f"{name} ({code}, {cur})"


COUNTRY_LABELS = [label_for("G163")] + sorted(
    (label_for(c) for c in COUNTRIES if c != "G163"), key=str.lower
) + [OTHER_LABEL]
LABEL_TO_CODE = {label_for(c): c for c in COUNTRIES}


# ---------------------------------------------------------------------------
# IMF data access
# ---------------------------------------------------------------------------

class IMFError(RuntimeError):
    """Raised when the IMF API cannot be reached or returns unusable data."""


def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    headers = {"User-Agent": "rer-explorer/1.0 (academic project)", "Cache-Control": "no-cache"}
    api_key = os.environ.get("IMF_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("IMF_API_KEY")  # type: ignore[attr-defined]
        except Exception:
            api_key = None
    if api_key:
        headers["Ocp-Apim-Subscription-Key"] = api_key
    s.headers.update(headers)
    return s


_TIME_RE = re.compile(r"^\s*(\d{4})\s*-?\s*M?\s*(\d{1,2})\s*$", re.IGNORECASE)


def parse_month(value) -> pd.Period | None:
    """Parse IMF monthly time codes such as '1976-M01', '1976-01' or '1976M1'."""
    m = _TIME_RE.match(str(value))
    if not m:
        return None
    year, month = int(m.group(1)), int(m.group(2))
    if not 1 <= month <= 12:
        return None
    return pd.Period(year=year, month=month, freq="M")


def _to_series(pairs: list[tuple[object, object]]) -> pd.Series:
    periods, values = [], []
    for t, v in pairs:
        p = parse_month(t)
        if p is None:
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(x):
            periods.append(p)
            values.append(x)
    if not periods:
        return pd.Series(dtype="float64", index=pd.PeriodIndex([], freq="M"))
    s = pd.Series(values, index=pd.PeriodIndex(periods, freq="M"), dtype="float64")
    s = s[~s.index.duplicated(keep="first")].sort_index()
    return s


def parse_sdmx_csv(text: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text), dtype=str)
    cols = {c.strip().upper(): c for c in df.columns}
    t_col, v_col = cols.get("TIME_PERIOD"), cols.get("OBS_VALUE")
    if t_col is None or v_col is None:
        raise IMFError("CSV response has no TIME_PERIOD / OBS_VALUE columns.")
    return _to_series(list(zip(df[t_col], df[v_col])))


def parse_sdmx_json(payload: dict) -> pd.Series:
    data = payload.get("data", payload)
    datasets = data.get("dataSets") or []
    structures = data.get("structures") or ([data["structure"]] if "structure" in data else [])
    if not datasets or not structures:
        return _to_series([])
    dims = structures[0].get("dimensions", {})
    pairs: list[tuple[object, object]] = []

    obs_dims = dims.get("observation", [])
    series_dims = dims.get("series", [])
    t_pos = next((i for i, d in enumerate(obs_dims) if d.get("id") == "TIME_PERIOD"), None)

    def time_value(dim: dict, idx: int):
        v = dim["values"][idx]
        return v.get("value", v.get("id")) if isinstance(v, dict) else v

    ds = datasets[0]
    if "series" in ds and t_pos is not None:
        for _, s in (ds.get("series") or {}).items():
            for k, obs in (s.get("observations") or {}).items():
                idx = int(str(k).split(":")[t_pos])
                val = obs[0] if isinstance(obs, list) else obs
                pairs.append((time_value(obs_dims[t_pos], idx), val))
            break  # one series expected for a fully specified key
    elif "observations" in ds:
        all_dims = series_dims + obs_dims
        t_all = next((i for i, d in enumerate(all_dims) if d.get("id") == "TIME_PERIOD"), None)
        if t_all is None:
            return _to_series([])
        for k, obs in ds["observations"].items():
            idx = int(str(k).split(":")[t_all])
            val = obs[0] if isinstance(obs, list) else obs
            pairs.append((time_value(all_dims[t_all], idx), val))
    return _to_series(pairs)


def parse_sdmx_xml(text: str) -> pd.Series:
    root = ET.fromstring(text)
    pairs = []
    for el in root.iter():
        a = el.attrib
        if "TIME_PERIOD" in a and "OBS_VALUE" in a:
            pairs.append((a["TIME_PERIOD"], a["OBS_VALUE"]))
    return _to_series(pairs)


def parse_response(text: str) -> pd.Series:
    """Detect the format of an SDMX response and return a monthly series."""
    body = text.lstrip("﻿ \r\n\t")
    if body.startswith("{"):
        return parse_sdmx_json(json.loads(body))
    if body.startswith("<"):
        return parse_sdmx_xml(body)
    return parse_sdmx_csv(body)


FORMAT_ATTEMPTS = (
    ("application/vnd.sdmx.data+csv;version=2.0.0", {}),
    ("application/vnd.sdmx.data+json;version=2.0.0", {"dimensionAtObservation": "TIME_PERIOD"}),
    ("application/vnd.sdmx.structurespecificdata+xml;version=2.1", {}),
)


def build_url(dataflow: str, key: str) -> str:
    return f"{IMF_BASE_URL}/data/dataflow/{IMF_AGENCY}/{dataflow}/+/{key}"


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_series(dataflow: str, key: str) -> tuple[pd.Series, str]:
    """Download one monthly series. Returns (series, url). Empty series = no data."""
    url = build_url(dataflow, key)
    session = _session()
    last_error = None
    for accept, params in FORMAT_ATTEMPTS:
        try:
            r = session.get(url, headers={"Accept": accept}, params=params, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            last_error = f"network error: {exc}"
            continue
        if r.status_code == 404:
            # SDMX returns 404 "NoResultsFound" when a key has no data.
            return _to_series([]), url
        if r.status_code in (406, 415):
            last_error = f"HTTP {r.status_code} for {accept}"
            continue
        if not r.ok:
            last_error = f"HTTP {r.status_code}: {r.text[:300]}"
            continue
        try:
            s = parse_response(r.text)
        except Exception as exc:  # malformed or unexpected format: try next one
            last_error = f"could not parse response ({type(exc).__name__}: {exc})"
            continue
        return s[s.index >= FIRST_MONTH], url
    raise IMFError(f"{url}\n{last_error}")


def usd_per_usd() -> pd.Series:
    """The US exchange rate against itself is 1 by definition."""
    idx = pd.period_range(FIRST_MONTH, pd.Period(date.today(), freq="M"), freq="M")
    return pd.Series(1.0, index=idx)


def load_country(code: str) -> dict:
    cpi, cpi_url = fetch_series(CPI_DATAFLOW, CPI_KEY.format(iso3=code))
    if code == "USA":
        er, er_url = usd_per_usd(), "(not requested: USD per USD = 1)"
    else:
        er, er_url = fetch_series(ER_DATAFLOW, ER_KEY.format(iso3=code))
    return {"cpi": cpi, "er": er, "urls": [cpi_url, er_url]}


# ---------------------------------------------------------------------------
# Calculations (pure functions, easy to test)
# ---------------------------------------------------------------------------

def build_frame(e1: pd.Series, e2: pd.Series, p1: pd.Series, p2: pd.Series) -> pd.DataFrame:
    """Combine raw inputs on a continuous monthly index and compute S and RER.

    e1, e2: local currency per USD for Country 1 and Country 2
    p1, p2: CPI indices for Country 1 and Country 2
    """
    df = pd.concat({"E1": e1, "E2": e2, "P1": p1, "P2": p2}, axis=1).sort_index()
    df = df[df.index >= FIRST_MONTH]
    if df.empty:
        return df
    full = pd.period_range(df.index.min(), df.index.max(), freq="M")
    df = df.reindex(full)
    df.index.name = "month"

    # S(t) = (C2 per USD) / (C1 per USD) = units of C2 per 1 unit of C1
    df["S"] = df["E2"] / df["E1"]
    df["valid"] = (
        df[["E1", "E2", "P1", "P2"]].notna().all(axis=1)
        & (df[["E1", "E2", "P1", "P2"]] > 0).all(axis=1)
    )
    df.loc[~df["valid"], "S"] = np.nan
    df["RER"] = df["S"] * df["P1"] / df["P2"]

    dlog = np.log(df["S"]).diff()
    df["possible_break"] = dlog.abs() > BREAK_LOG_THRESHOLD
    return df


def compute_sample(df: pd.DataFrame, start: pd.Period, end: pd.Period) -> tuple[pd.DataFrame, float]:
    """Slice to [start, end] and compute RER*, Real Index, deviation and PPP rate."""
    s = df.loc[start:end].copy()
    rer_star = float(s.loc[s["valid"], "RER"].mean())
    s["RER_star"] = rer_star
    s["real_index"] = 100.0 * s["RER"] / rer_star
    s["deviation_pct"] = s["real_index"] - 100.0
    s["S_PPP"] = rer_star * s["P2"] / s["P1"]
    s.loc[~s["valid"], ["real_index", "deviation_pct", "S_PPP"]] = np.nan
    return s, rer_star


def half_life(rer: pd.Series) -> dict | None:
    """AR(1) on ln RER using consecutive valid months: q_t = a + rho * q_{t-1} + e_t."""
    q = np.log(rer.astype("float64"))
    lag = q.shift(1)
    mask = q.notna() & lag.notna()
    n = int(mask.sum())
    if n < MIN_OBS_HALF_LIFE:
        return None
    X = np.column_stack([np.ones(n), lag[mask].to_numpy()])
    y = q[mask].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    rho = float(beta[1])
    if 0 < rho < 1:
        hl_months = float(np.log(0.5) / np.log(rho))
    else:
        hl_months = float("inf")
    return {"rho": rho, "months": hl_months, "years": hl_months / 12.0, "n": n}


def fmt_num(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    ax = abs(x)
    if ax >= 1000:
        return f"{x:,.1f}"
    if ax >= 1:
        return f"{x:,.4f}"
    return f"{x:.6g}"


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _dates(s: pd.DataFrame):
    return s.index.to_timestamp()


def add_shading(fig: go.Figure, s: pd.DataFrame, secondary: bool | None = None) -> None:
    x = _dates(s)
    idx = s["real_index"]
    above = idx.where(idx.isna(), np.maximum(idx, 100.0))
    below = idx.where(idx.isna(), np.minimum(idx, 100.0))
    base = idx.where(idx.isna(), 100.0)
    kw = {} if secondary is None else {"secondary_y": secondary}
    for series, color, name in ((above, FILL_OVER, "Overvalued"), (below, FILL_UNDER, "Undervalued")):
        fig.add_trace(go.Scatter(x=x, y=base, mode="lines", line=dict(width=0),
                                 hoverinfo="skip", showlegend=False), **kw)
        fig.add_trace(go.Scatter(x=x, y=series, mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=color, name=name, hoverinfo="skip"), **kw)


def real_trace(s: pd.DataFrame) -> go.Scatter:
    return go.Scatter(
        x=_dates(s), y=s["real_index"], mode="lines", name="Real Index (RER* = 100)",
        line=dict(color=COLOR_INDEX, width=2.2),
        hovertemplate="%{x|%b %Y}<br>Real Index: %{y:.1f}<extra></extra>",
    )


def nominal_traces(s: pd.DataFrame, cur1: str, cur2: str, show_ppp: bool) -> list[go.Scatter]:
    unit = f"{cur2} per {cur1}"
    traces = [go.Scatter(
        x=_dates(s), y=s["S"], mode="lines", name=f"Nominal S ({unit})",
        line=dict(color=COLOR_NOMINAL, width=2),
        hovertemplate="%{x|%b %Y}<br>S: %{y:,.6g} " + unit + "<extra></extra>",
    )]
    if show_ppp:
        traces.append(go.Scatter(
            x=_dates(s), y=s["S_PPP"], mode="lines", name="PPP-implied S at RER*",
            line=dict(color=COLOR_PPP, width=1.6, dash="dash"),
            hovertemplate="%{x|%b %Y}<br>PPP-implied S: %{y:,.6g} " + unit + "<extra></extra>",
        ))
    return traces


def make_chart(s: pd.DataFrame, view: str, cur1: str, cur2: str, log_nominal: bool,
               shade: bool, show_ppp: bool) -> go.Figure:
    unit = f"{cur2} per 1 {cur1}"
    layout = dict(
        height=560, hovermode="x unified", template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    if view == "Real":
        fig = go.Figure()
        if shade:
            add_shading(fig, s)
        fig.add_trace(real_trace(s))
        fig.add_hline(y=100, line=dict(color="#333", width=1, dash="dot"))
        fig.update_yaxes(title_text="Real Index (sample average = 100)")
    elif view == "Nominal":
        fig = go.Figure(nominal_traces(s, cur1, cur2, show_ppp))
        fig.update_yaxes(title_text=f"S(t): {unit}", type="log" if log_nominal else "linear")
    else:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        if shade:
            add_shading(fig, s, secondary=False)
        fig.add_trace(real_trace(s), secondary_y=False)
        for t in nominal_traces(s, cur1, cur2, show_ppp):
            fig.add_trace(t, secondary_y=True)
        fig.add_hline(y=100, line=dict(color="#333", width=1, dash="dot"), secondary_y=False)
        fig.update_yaxes(title_text="Real Index (sample average = 100)",
                         title_font=dict(color=COLOR_INDEX), secondary_y=False)
        fig.update_yaxes(title_text=f"S(t): {unit}", title_font=dict(color=COLOR_NOMINAL),
                         type="log" if log_nominal else "linear", showgrid=False, secondary_y=True)
    fig.update_layout(**layout)
    fig.update_xaxes(title_text=None)
    return fig


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def init_state() -> None:
    st.session_state.setdefault("c1", label_for("MEX"))
    st.session_state.setdefault("c2", label_for("USA"))
    st.session_state.setdefault("c1_other", "")
    st.session_state.setdefault("c2_other", "")


def swap_countries() -> None:
    ss = st.session_state
    ss["c1"], ss["c2"] = ss["c2"], ss["c1"]
    ss["c1_other"], ss["c2_other"] = ss["c2_other"], ss["c1_other"]


def apply_preset() -> None:
    pair = PRESETS.get(st.session_state.get("preset"))
    if pair:
        st.session_state["c1"] = label_for(pair[0])
        st.session_state["c2"] = label_for(pair[1])
    st.session_state["preset"] = "Choose a preset..."


def resolve_country(label: str, other: str) -> tuple[str | None, str, str]:
    """Return (code, display name, currency label)."""
    if label != OTHER_LABEL:
        code = LABEL_TO_CODE[label]
        name, cur = COUNTRIES[code]
        return code, name, cur
    code = other.strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{3,4}", code):
        return None, "", ""
    if code in COUNTRIES:
        name, cur = COUNTRIES[code]
        return code, name, cur
    return code, code, f"{code} currency"


def sidebar() -> dict:
    sb = st.sidebar
    sb.header("Currency pair")
    sb.selectbox("Quick preset", list(PRESETS), key="preset", on_change=apply_preset)
    sb.selectbox("Country 1 (base currency)", COUNTRY_LABELS, key="c1")
    if st.session_state["c1"] == OTHER_LABEL:
        sb.text_input("Country 1 ISO3 code", key="c1_other", placeholder="e.g. MEX")
    sb.button("Swap countries", on_click=swap_countries, width="stretch")
    sb.selectbox("Country 2 (quote currency)", COUNTRY_LABELS, key="c2")
    if st.session_state["c2"] == OTHER_LABEL:
        sb.text_input("Country 2 ISO3 code", key="c2_other", placeholder="e.g. USA")

    sb.header("Chart")
    view = sb.radio("Series to plot", ["Real", "Nominal", "Both"], index=2, horizontal=True)
    log_nominal = sb.checkbox("Log scale for nominal S", value=False,
                              help="Recommended for high-inflation currencies, where S spans "
                                   "several orders of magnitude.")
    shade = sb.checkbox("Shade over/undervaluation", value=True)
    show_ppp = sb.checkbox("Show PPP-implied exchange rate", value=True)
    sb.caption("Data: IMF SDMX 3.0 API (CPI and ER dataflows). Cached for 24 hours.")
    if sb.button("Clear cached IMF data", width="stretch"):
        st.cache_data.clear()
    return {"view": view, "log_nominal": log_nominal, "shade": shade, "show_ppp": show_ppp}


def main() -> None:
    st.set_page_config(page_title="Real Exchange Rate Explorer", page_icon="📈", layout="wide")
    init_state()
    opts = sidebar()

    st.title("Real Exchange Rate Explorer")
    st.caption("S(t) = units of Country 2 currency per 1 unit of Country 1 currency.  "
               "RER(t) = S(t) × P1(t) / P2(t).  Real Index(t) = 100 × RER(t) / RER*.")

    code1, name1, cur1 = resolve_country(st.session_state["c1"], st.session_state["c1_other"])
    code2, name2, cur2 = resolve_country(st.session_state["c2"], st.session_state["c2_other"])
    if not code1 or not code2:
        st.info("Type a valid IMF ISO3 code for the country marked 'Other'.")
        st.stop()
    if code1 == code2:
        st.warning("Choose two different countries.")
        st.stop()
    if cur1 == cur2 and cur1 not in ("", None):
        st.info(f"Both countries use {cur1}, so nominal S(t) is 1 whenever both share the currency; "
                "the real index then reflects relative inflation only.")

    try:
        with st.spinner(f"Downloading IMF data for {name1} and {name2}..."):
            d1 = load_country(code1)
            d2 = load_country(code2)
    except IMFError as exc:
        st.error("Could not retrieve data from the IMF API. It may be temporarily down or "
                 "rate limiting. Try again in a minute, or use 'Clear cached IMF data'.")
        with st.expander("Technical details"):
            st.code(str(exc))
        st.stop()

    raw = {
        f"CPI {name1} (P1)": d1["cpi"], f"CPI {name2} (P2)": d2["cpi"],
        f"{cur1} per USD (Country 1)": d1["er"], f"{cur2} per USD (Country 2)": d2["er"],
    }
    missing = [k for k, v in raw.items() if v.dropna().empty]
    if missing:
        st.error("The IMF returned no monthly data for: " + "; ".join(missing) +
                 ". Coverage varies by country; try another pair.")
        with st.expander("Requested URLs"):
            for u in d1["urls"] + d2["urls"]:
                st.code(u)
        st.stop()

    df = build_frame(d1["er"], d2["er"], d1["cpi"], d2["cpi"])
    valid_months = df.index[df["valid"]]
    if len(valid_months) < 2:
        st.error("There are fewer than two months where both CPIs and both exchange rates "
                 "overlap. Try another pair.")
        st.stop()

    # Sample selection (defaults to the full valid range)
    options = [str(p) for p in pd.period_range(valid_months.min(), valid_months.max(), freq="M")]
    range_key = f"range_{code1}_{code2}"
    current = st.session_state.get(range_key)
    if current is not None and (current[0] not in options or current[1] not in options):
        del st.session_state[range_key]
    start_str, end_str = st.select_slider(
        "Sample period (RER* is the average RER over valid months in this window)",
        options=options, value=(options[0], options[-1]), key=range_key,
    )
    start, end = pd.Period(start_str, freq="M"), pd.Period(end_str, freq="M")
    sample, rer_star = compute_sample(df, start, end)
    n_valid = int(sample["valid"].sum())
    if n_valid == 0 or not np.isfinite(rer_star):
        st.warning("No valid months in the selected window.")
        st.stop()

    # --- Valuation readout -------------------------------------------------
    last = sample[sample["valid"]].iloc[-1]
    last_month = sample[sample["valid"]].index[-1]
    dev = float(last["deviation_pct"])
    direction = "overvalued" if dev > 0 else "undervalued"
    hl = half_life(sample["RER"])

    st.markdown(
        f"**In {last_month.strftime('%b %Y')}, the {cur1} is {abs(dev):.1f}% {direction} against "
        f"the {cur2}** relative to its {start.strftime('%b %Y')} to {end.strftime('%b %Y')} average "
        f"real exchange rate (Real Index = {last['real_index']:.1f})."
    )
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Real Index (latest)", f"{last['real_index']:.1f}", f"{dev:+.1f}% vs RER*",
              delta_color="off")
    m2.metric(f"Nominal S ({cur2} per {cur1})", fmt_num(float(last["S"])))
    m3.metric("PPP-implied S at RER*", fmt_num(float(last["S_PPP"])))
    m4.metric("RER*", fmt_num(rer_star))
    if hl is None:
        m5.metric("Half-life of deviations", "n/a", f"needs {MIN_OBS_HALF_LIFE}+ months",
                  delta_color="off")
    elif np.isinf(hl["months"]):
        m5.metric("Half-life of deviations", "no reversion", f"rho = {hl['rho']:.3f}",
                  delta_color="off")
    else:
        m5.metric("Half-life of deviations", f"{hl['years']:.1f} years",
                  f"rho = {hl['rho']:.3f}, n = {hl['n']}", delta_color="off")

    # --- Chart ---------------------------------------------------------------
    fig = make_chart(sample, opts["view"], cur1, cur2, opts["log_nominal"], opts["shade"],
                     opts["show_ppp"])
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Valid months in sample: {n_valid} of {len(sample)}. "
               "Index above 100: Country 1's currency is real appreciated (overvalued) relative "
               "to the sample average. Below 100: undervalued.")

    # --- Data quality warnings -------------------------------------------------
    breaks = sample.index[sample["possible_break"].fillna(False).astype(bool)]
    if len(breaks):
        months = ", ".join(p.strftime("%b %Y") for p in breaks[:12])
        more = "" if len(breaks) <= 12 else f" and {len(breaks) - 12} more"
        st.warning(
            f"Possible series breaks: S(t) moved by more than 50% in one month in {months}{more}. "
            "This can be a genuine crisis devaluation or a currency redenomination / legacy "
            "currency change in the IMF series. Check before interpreting levels across it."
        )
    for code, name in ((code1, name1), (code2, name2)):
        adopt = EURO_ADOPTION.get(code)
        if adopt and start < pd.Period(adopt, freq="M"):
            st.info(f"{name} adopted the euro in {pd.Period(adopt, freq='M').strftime('%b %Y')}. "
                    "Before then its IMF exchange rate may be in the legacy currency, or "
                    "coverage may be limited (Euro area series), so check for a level break.")

    with st.expander("Data coverage"):
        cov_raw = {
            f"CPI {name1} (P1)": (d1["cpi"], "P1"), f"CPI {name2} (P2)": (d2["cpi"], "P2"),
            f"{cur1} per USD": (d1["er"], "E1"), f"{cur2} per USD": (d2["er"], "E2"),
        }
        rows = []
        for label, (s_raw, col) in cov_raw.items():
            s_raw = s_raw.dropna()
            rows.append({
                "Series": label,
                "First month": str(s_raw.index.min()) if len(s_raw) else "no data",
                "Last month": str(s_raw.index.max()) if len(s_raw) else "no data",
                "Observations (all)": int(len(s_raw)),
                "Missing in sample": int(sample[col].isna().sum()),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Months with any missing input are excluded from RER* and left blank on the "
                   "chart. Nothing is interpolated.")
        for u in d1["urls"] + d2["urls"]:
            st.code(u, language=None)

    # --- CSV download ------------------------------------------------------------
    out = sample.reset_index()
    out["month"] = out["month"].astype(str)
    out = out.rename(columns={
        "month": "month",
        "E1": f"{cur1}_per_USD",
        "E2": f"{cur2}_per_USD",
        "P1": f"CPI_{code1}",
        "P2": f"CPI_{code2}",
        "S": f"S_{cur2}_per_{cur1}",
        "valid": "valid_month",
        "RER": "RER",
        "RER_star": "RER_star",
        "real_index": "real_index",
        "deviation_pct": "deviation_vs_RER_star_pct",
        "S_PPP": f"S_PPP_{cur2}_per_{cur1}",
        "possible_break": "possible_break_flag",
    })
    cols = ["month", f"{cur1}_per_USD", f"{cur2}_per_USD", f"CPI_{code1}", f"CPI_{code2}",
            f"S_{cur2}_per_{cur1}", "RER", "RER_star", "real_index", "deviation_vs_RER_star_pct",
            f"S_PPP_{cur2}_per_{cur1}", "valid_month", "possible_break_flag"]
    out = out[cols]
    st.download_button(
        "Download CSV of the selected sample",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name=f"RER_{code1}_{code2}_{start}_{end}.csv",
        mime="text/csv",
    )
    with st.expander("Preview data table"):
        st.dataframe(out, hide_index=True, width="stretch")

    # --- Methodology -------------------------------------------------------------
    with st.expander("Methodology"):
        st.markdown(f"""
**Data.** Monthly CPI (`{{ISO3}}.CPI._T.IX.M`) and monthly average exchange rates
(`{{ISO3}}.XDC_USD.PA_RT.M`) from the IMF SDMX 3.0 API, from January 1976 where available.
The Euro area uses code G163. The US rate against the dollar is set to 1.

**Nominal rate.** The IMF reports local currency per USD (E). Converting to the stated direction:

S(t) = E2(t) / E1(t) = units of {cur2} per 1 {cur1}

**Real exchange rate.** RER(t) = S(t) × P1(t) / P2(t). It measures the price of Country 1's
basket in terms of Country 2's basket. A rise is a real appreciation of the {cur1}.

**Benchmark.** RER* = average of RER(t) over valid months in the selected window (months where
both CPIs and both exchange rates exist). Real Index(t) = 100 × RER(t) / RER*. Because CPI base
years cancel out in this ratio, the index does not depend on how each CPI is rebased.

**PPP-implied rate.** S_PPP(t) = RER* × P2(t) / P1(t): the nominal rate that would put the RER
exactly at its sample average. Actual S above S_PPP means the {cur1} is overvalued.

**Half-life.** OLS estimate of ln RER(t) = a + ρ ln RER(t−1) + e(t) on consecutive valid months.
Half-life = ln(0.5) / ln(ρ) months, reported in years. The PPP literature typically finds 3 to 5 years.

**Caveat.** The benchmark is relative PPP against the chosen sample average, so conclusions
depend on the window. Structural shifts (e.g. Balassa-Samuelson productivity effects) can move the
equilibrium RER over time.
""")


if __name__ == "__main__":
    main()
