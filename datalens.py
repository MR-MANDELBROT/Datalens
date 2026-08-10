"""
DataLens – Interaktive Datenanalyse
====================================
    pip install dash pandas plotly scipy openpyxl
    python datalens.py
"""

import base64
import io
import colorsys
import hashlib
import re

import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html, dash_table, no_update
from dash.exceptions import PreventUpdate
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import pdist

# =============================================================================
# Internal column names after melt
# =============================================================================
COL_NAME  = "Spaltenname"
COL_VALUE = "Spaltenwert"
COL_COUNT = "Berechneter Wert"

VIRTUAL_LABELS = {
    COL_NAME:  "Spaltenname  (welche der verglichenen Spalten)",
    COL_VALUE: "Spaltenwert  (der Wert in der jeweiligen Spalte)",
    COL_COUNT: "Berechneter Wert  (Ergebnis der Aggregation)",
}

_HEX_RE = re.compile(r'^#[0-9a-fA-F]{3,8}$')

# =============================================================================
# Color palettes
# =============================================================================

PALETTES = {
    "standard": {
        "label": "Standard (Golden Angle)",
        "colors": None,  # uses generate_distinct_colors
    },
    "plotly": {
        "label": "Plotly",
        "colors": px.colors.qualitative.Plotly,
    },
    "pastel": {
        "label": "Pastell",
        "colors": ["#a1c9f4", "#ffb482", "#8de5a1", "#ff9f9b", "#d0bbff",
                    "#debb9b", "#fab0e4", "#cfcfcf", "#fffea3", "#b9f2f0",
                    "#f4a6d7", "#b5e6a3", "#f9c784", "#a3d5f7", "#e6a3a3"],
    },
    "bold": {
        "label": "Kraeftig",
        "colors": ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
                    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
                    "#469990", "#dcbeff", "#9A6324", "#800000", "#aaffc3"],
    },
    "earth": {
        "label": "Erdtoene",
        "colors": ["#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
                    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"],
    },
    "monochrome": {
        "label": "Monochrom (Blau)",
        "colors": ["#08306b", "#08519c", "#2171b5", "#4292c6", "#6baed6",
                    "#9ecae1", "#c6dbef", "#deebf7", "#3182bd", "#084594",
                    "#0d47a1", "#1565c0", "#1976d2", "#1e88e5", "#2196f3"],
    },
}


def generate_distinct_colors(n):
    if n <= 0:
        return []
    colors = []
    golden = 0.618033988749895
    for i in range(n):
        h = (i * golden) % 1.0
        s = 0.65 + 0.2 * (i % 3) / 2
        l_val = 0.50 + 0.12 * ((i // 3) % 3 - 1)
        r, g, b = colorsys.hls_to_rgb(h, l_val, s)
        colors.append(f"rgb({int(r*255)},{int(g*255)},{int(b*255)})")
    return colors


def create_color_mapping(values, palette_name="standard"):
    unique = sorted(set(str(v) for v in values if pd.notna(v)))
    n = len(unique)
    if n == 0:
        return {}

    pal_info = PALETTES.get(palette_name, PALETTES["standard"])
    if pal_info["colors"] is None:
        palette = generate_distinct_colors(max(n, 12))
    else:
        base = list(pal_info["colors"])
        # Extend if needed
        while len(base) < n:
            base += base
        palette = base

    mapping = {}
    for val in unique:
        idx = int(hashlib.md5(str(val).encode()).hexdigest(), 16) % len(palette)
        attempts = 0
        while palette[idx] in mapping.values() and attempts < len(palette):
            idx = (idx + 1) % len(palette)
            attempts += 1
        mapping[val] = palette[idx]
    return mapping


# =============================================================================
# CSS
# =============================================================================

APP_CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { box-sizing: border-box; }
body { margin: 0; padding: 0; }

.light {
    --bg: #f4f6f9; --card: #ffffff; --text: #1e293b; --text-sec: #64748b;
    --text-muted: #94a3b8; --border: #e2e8f0; --input-bg: #ffffff;
    --input-border: #cbd5e1; --hover: #f1f5f9; --accent: #3b82f6;
    --shadow: 0 1px 3px rgba(0,0,0,0.08); --upload-bg: #f8fafc;
    --tag-bg: #e0f2fe; --tag-text: #0369a1; --tag-bg-num: #dcfce7;
    --tag-text-num: #15803d; --stripe: #fafbfc; --th-bg: #f8fafc;
}
.dark {
    --bg: #0b1120; --card: #151d2e; --text: #e2e8f0; --text-sec: #94a3b8;
    --text-muted: #64748b; --border: #1e2d44; --input-bg: #1a2538;
    --input-border: #2a3a52; --hover: #1e2d44; --accent: #60a5fa;
    --shadow: 0 1px 4px rgba(0,0,0,0.3); --upload-bg: #111827;
    --tag-bg: #1e3a5f; --tag-text: #7dd3fc; --tag-bg-num: #14532d;
    --tag-text-num: #86efac; --stripe: #111827; --th-bg: #1a2538;
}

.app-container {
    font-family: 'Inter', -apple-system, system-ui, sans-serif;
    background: var(--bg); color: var(--text);
    min-height: 100vh; padding: 24px 40px 60px;
    transition: background 0.25s, color 0.25s;
}
@media (max-width: 768px) { .app-container { padding: 16px; } }

.top-bar { display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px; }
.top-bar h1 { font-size: 26px; font-weight: 700; margin: 0; }
.top-bar p  { font-size: 13px; color: var(--text-sec); margin: 2px 0 0; }

.theme-btn {
    background: var(--card); border: 1px solid var(--border); border-radius: 8px;
    padding: 8px 14px; cursor: pointer; font-size: 14px; color: var(--text);
    transition: background 0.2s;
}
.theme-btn:hover { background: var(--hover); }

.card {
    background: var(--card); border: 1px solid var(--border); border-radius: 12px;
    padding: 24px; margin-bottom: 20px; box-shadow: var(--shadow);
    transition: background 0.25s, border-color 0.25s;
}

.section-title {
    font-size: 15px; font-weight: 600; color: var(--text);
    margin: 0 0 16px; padding-bottom: 8px; border-bottom: 2px solid var(--border);
    cursor: pointer; user-select: none; list-style: none;
    display: flex; align-items: center; gap: 6px;
}
.section-title::-webkit-details-marker { display: none; }
.section-title::before { content: "\25B6"; font-size: 10px; transition: transform 0.2s; }
details[open] > .section-title::before { transform: rotate(90deg); }

.field-label { font-size: 13px; font-weight: 500; color: var(--text-sec); margin-bottom: 4px; display: block; }
.help-text { font-size: 11px; color: var(--text-muted); margin: 2px 0 12px; line-height: 1.4; }

.two-col { display: flex; gap: 20px; flex-wrap: wrap; }
.two-col > .card { flex: 1 1 420px; min-width: 0; }

.upload-zone {
    border: 2px dashed var(--border); border-radius: 12px; padding: 48px 20px;
    text-align: center; background: var(--upload-bg); cursor: pointer; transition: border-color 0.2s;
}
.upload-zone:hover { border-color: var(--accent); }
.upload-icon { font-size: 32px; margin-bottom: 8px; }
.upload-main { font-size: 15px; color: var(--text); font-weight: 500; }
.upload-sub  { font-size: 12px; color: var(--text-muted); margin-top: 4px; }

.col-tag { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 500; margin: 2px 4px; }
.col-tag.cat { background: var(--tag-bg); color: var(--tag-text); }
.col-tag.num { background: var(--tag-bg-num); color: var(--tag-text-num); }

.chart-warning { padding: 10px 14px; border-radius: 8px; margin-bottom: 14px; font-size: 13px; line-height: 1.5; }
.chart-warning.warn { background: #fef3c7; border: 1px solid #f59e0b; color: #92400e; }
.dark .chart-warning.warn { background: #422006; border-color: #b45309; color: #fde68a; }

.inline-fields { display: flex; gap: 12px; flex-wrap: wrap; }
.inline-fields > div { flex: 1 1 140px; min-width: 120px; }

.palette-preview { display: flex; gap: 3px; margin-top: 4px; margin-bottom: 10px; }
.palette-swatch { width: 20px; height: 16px; border-radius: 3px; }

/* DataTable dark */
.dark .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner td {
    background-color: #151d2e !important; color: #e2e8f0 !important; border-color: #1e2d44 !important;
}
.dark .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th,
.dark .dash-table-container .dash-header {
    background-color: #1a2538 !important; color: #e2e8f0 !important; border-color: #1e2d44 !important;
}
.dark .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner tr:nth-child(odd) td {
    background-color: #111827 !important;
}
.dark .dash-table-container .previous-next-container { color: #e2e8f0 !important; }

/* Text inputs dark */
.dark input[type="text"], .dark input[type="number"] {
    background-color: #1a2538 !important; border-color: #2a3a52 !important; color: #e2e8f0 !important;
}
"""

# JS that forces dropdown styling – runs on every theme change
# =============================================================================
# Data helpers
# =============================================================================

def parse_upload(contents, filename):
    if contents is None:
        return None
    try:
        _, content_string = contents.split(",", 1)
    except ValueError:
        return None
    decoded = base64.b64decode(content_string)
    try:
        if filename.lower().endswith(".csv"):
            return pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        if filename.lower().endswith((".xls", ".xlsx")):
            return pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(f"[FEHLER] {e}")
    return None


def classify_columns(df):
    result = {}
    for col in df.columns:
        ratio = pd.to_numeric(df[col], errors="coerce").notna().mean()
        result[col] = "numeric" if ratio > 0.8 else "categorical"
    return result


def safe_melt(df, id_vars, value_vars):
    existing = [c for c in value_vars if c in df.columns]
    if not existing:
        return pd.DataFrame()
    valid_ids = [c for c in id_vars if c in df.columns and c not in existing]
    melted = df.melt(id_vars=valid_ids, value_vars=existing, var_name=COL_NAME, value_name=COL_VALUE)
    return melted.dropna(subset=[COL_VALUE])


def safe_pivot(grouped, index_col, columns_col, value_col):
    if index_col == columns_col:
        return None, "X und Y Achse duerfen nicht identisch sein."
    try:
        return grouped.pivot(index=index_col, columns=columns_col, values=value_col).fillna(0), None
    except ValueError as e:
        if "duplicate" in str(e).lower():
            agg = grouped.groupby([index_col, columns_col], as_index=False)[value_col].sum()
            return agg.pivot(index=index_col, columns=columns_col, values=value_col).fillna(0), "aggregated"
        return None, str(e)


def is_numeric(s):
    return pd.to_numeric(s, errors="coerce").notna().all()


def sort_by_distribution(grouped, dim, other_dims, metric="euclidean", method="average", normalize=True):
    if dim not in grouped.columns:
        return []
    available = [d for d in other_dims if d in grouped.columns]
    if not available:
        return sorted(grouped[dim].unique())
    pivot_df, _ = safe_pivot(grouped, index_col=dim, columns_col=available[0], value_col=COL_COUNT)
    if pivot_df is None or pivot_df.shape[0] < 2:
        return sorted(grouped[dim].unique()) if pivot_df is None else list(pivot_df.index)
    mat = pivot_df.values.astype(float)
    if normalize:
        rs = mat.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1
        mat = mat / rs
    try:
        dist = pdist(mat, metric=metric)
        linked = optimal_leaf_ordering(linkage(dist, method=method), dist)
        return list(pivot_df.index[leaves_list(linked)])
    except Exception:
        return sorted(pivot_df.index)


def sort_dimension(grouped, col, mode, other_dims=None):
    if col not in grouped.columns:
        return []
    vals = grouped[col].unique()
    if mode == "alphabetical":
        return sorted(vals)
    if mode in ("sum_asc", "sum_desc"):
        sums = grouped.groupby(col)[COL_COUNT].sum()
        return list(sums.sort_values(ascending=(mode == "sum_asc")).index)
    if mode == "similarity":
        return sort_by_distribution(grouped, col, other_dims or [])
    return sorted(vals)


def build_groupby_keys(fixed_cols, x_axis, color_col):
    keys = list(fixed_cols)
    virt = {COL_NAME, COL_VALUE}
    if x_axis in virt:
        keys.append(x_axis)
        other = (virt - {x_axis}).pop()
        if color_col == other:
            keys.append(other)
    else:
        keys.extend([COL_NAME, COL_VALUE])
    if color_col and color_col not in virt and color_col not in keys:
        keys.append(color_col)
    seen = set()
    return [k for k in keys if not (k in seen or seen.add(k))]


def aggregate_data(melted, aggfunc, x_axis, color_col, fixed_cols):
    keys = build_groupby_keys(fixed_cols, x_axis, color_col)
    keys = [k for k in keys if k in melted.columns]
    if not keys:
        return pd.DataFrame()
    if aggfunc == "count":
        return melted.groupby(keys).size().reset_index(name=COL_COUNT)
    if aggfunc in ("sum", "mean"):
        if not is_numeric(melted[COL_VALUE]):
            return melted.groupby(keys).size().reset_index(name=COL_COUNT)
        m2 = melted.copy()
        m2[COL_VALUE] = pd.to_numeric(m2[COL_VALUE], errors="coerce")
        return m2.groupby(keys)[COL_VALUE].agg(aggfunc).reset_index(name=COL_COUNT)
    if aggfunc == "percentage":
        base = list(fixed_cols) + [COL_NAME, COL_VALUE]
        if color_col and color_col not in base:
            base.append(color_col)
        base = [k for k in base if k in melted.columns]
        if not base:
            return pd.DataFrame()
        temp = melted.groupby(base, as_index=False).size().rename(columns={"size": "_cnt"})
        norm = [x_axis] if x_axis in temp.columns else [k for k in fixed_cols if k in temp.columns]
        if norm:
            totals = temp.groupby(norm, as_index=False)["_cnt"].sum().rename(columns={"_cnt": "_tot"})
            temp = temp.merge(totals, on=norm, how="left")
            temp[COL_COUNT] = (temp["_cnt"] / temp["_tot"].replace(0, 1)) * 100
        else:
            total = max(temp["_cnt"].sum(), 1)
            temp[COL_COUNT] = (temp["_cnt"] / total) * 100
        return temp.drop(columns=["_cnt", "_tot"], errors="ignore")
    return melted.groupby(keys).size().reset_index(name=COL_COUNT)


def fmt(x, aggfunc):
    if aggfunc == "percentage":
        return f"{x:.1f}%"
    if aggfunc in ("count", "sum"):
        return f"{int(x)}"
    if aggfunc == "mean":
        return f"{x:.2f}"
    return str(x)


def empty_figure(msg, theme):
    bg = "#ffffff" if theme == "light" else "#151d2e"
    c = "#94a3b8" if theme == "light" else "#475569"
    fig = go.Figure()
    fig.update_layout(
        annotations=[dict(text=msg, showarrow=False, font=dict(size=15, color=c))],
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        paper_bgcolor=bg, plot_bgcolor=bg,
    )
    return fig


def themed_layout(fig, title, xlab, ylab, theme, font_size=13, font_color=None,
                   chart_bg=None, grid_visible=True):
    dark = theme == "dark"
    default_bg  = "#151d2e" if dark else "#ffffff"
    default_txt = "#e2e8f0" if dark else "#1e293b"
    bg   = chart_bg or default_bg
    txt  = font_color or default_txt
    grid = "#1e2d44" if dark else "#f0f0f0"
    line = "#2a3a52" if dark else "#d1d5db"
    grid_c = grid if grid_visible else "rgba(0,0,0,0)"
    legend = (
        dict(orientation="v", x=1.02, y=1, bordercolor=line, borderwidth=1,
             bgcolor="rgba(0,0,0,0)" if dark else "rgba(255,255,255,0.9)")
        if title
        else dict(orientation="h", x=0.5, y=1.12, xanchor="center")
    )
    fig.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        autosize=True,
        margin=dict(l=60, r=40, t=70 if title else 44, b=90),
        font=dict(family="Inter, sans-serif", size=font_size, color=txt),
        title=dict(text=title, x=0.5, xanchor="center",
                   font=dict(size=font_size + 7)) if title else None,
        xaxis=dict(title=dict(text=xlab, font=dict(size=font_size + 1)),
                   automargin=True, tickangle=45, tickfont=dict(size=max(font_size - 2, 9)),
                   showline=True, linewidth=1, linecolor=line,
                   gridcolor=grid_c, gridwidth=0.5, zeroline=False),
        yaxis=dict(title=dict(text=ylab, font=dict(size=font_size + 1)),
                   automargin=True, tickfont=dict(size=max(font_size - 2, 9)),
                   showline=True, linewidth=1, linecolor=line,
                   gridcolor=grid_c, gridwidth=0.5, zeroline=False),
        legend=legend, legend_title_text="Legende",
        paper_bgcolor=bg, plot_bgcolor=bg,
        hoverlabel=dict(font_size=font_size - 1),
    )
    return fig


def build_warnings(plot_type, color_col, grouped, melted, aggfunc, col_types, value_cols):
    warnings = []
    if color_col and color_col in grouped.columns:
        n = grouped[color_col].nunique()
        if n > 20:
            warnings.append(
                f"{n} verschiedene Farbwerte. Das Diagramm wird schwer lesbar. "
                f"Eventuell andere Farbspalte oder weniger Daten waehlen."
            )
    if plot_type in ("bar", "pie") and color_col == COL_VALUE:
        num_vals = [c for c in (value_cols or []) if col_types.get(c) == "numeric"]
        if num_vals:
            warnings.append(
                f"Numerische Spalten ({', '.join(num_vals)}) als Farbkategorien. "
                f"Boxplot oder Scatter waere hier meist besser."
            )
    if aggfunc == "percentage" and color_col == COL_VALUE:
        if COL_VALUE in grouped.columns and grouped[COL_VALUE].nunique() > 30:
            warnings.append("Prozent-Modus mit sehr vielen Werten. Count oder Boxplot waere sinnvoller.")
    return warnings


# =============================================================================
# Dash App
# =============================================================================
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "DataLens"
app.index_string = (
    '<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}'
    '<style>' + APP_CSS + '</style></head>'
    '<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer>'
    '<script>'
    'function _dlStyle(){'
    '  var c=document.getElementById("app-container");'
    '  if(!c)return;'
    '  var dk=c.classList.contains("dark");'
    '  var bg=dk?"#1a2538":"#fff",bd=dk?"#2a3a52":"#cbd5e1",tx=dk?"#e2e8f0":"#1e293b",'
    '      hv=dk?"#1e2d44":"#f1f5f9",mt=dk?"#64748b":"#94a3b8";'
    '  document.querySelectorAll(".Select-control").forEach(function(e){'
    '    e.style.cssText="background-color:"+bg+"!important;border-color:"+bd+"!important;color:"+tx+"!important";'
    '  });'
    '  document.querySelectorAll(".Select-value-label,.Select-placeholder,.Select-input input,.Select--multi .Select-value-label").forEach(function(e){'
    '    e.style.color=tx;'
    '  });'
    '  document.querySelectorAll(".Select-placeholder").forEach(function(e){e.style.color=mt;});'
    '  document.querySelectorAll(".Select--multi .Select-value").forEach(function(e){'
    '    e.style.cssText="background-color:"+hv+"!important;border-color:"+bd+"!important;color:"+tx+"!important";'
    '  });'
    '  document.querySelectorAll(".Select-menu-outer").forEach(function(e){'
    '    e.style.cssText="background-color:"+bg+"!important;border-color:"+bd+"!important";'
    '  });'
    '  document.querySelectorAll(".Select-option").forEach(function(e){'
    '    e.style.cssText="background-color:"+bg+"!important;color:"+tx+"!important";'
    '  });'
    '  document.querySelectorAll(".Select-arrow").forEach(function(e){e.style.borderTopColor=mt;});'
    '  document.querySelectorAll(".Select-clear").forEach(function(e){e.style.color=mt;});'
    '  document.querySelectorAll(".Select-noresults").forEach(function(e){'
    '    e.style.cssText="background-color:"+bg+"!important;color:"+mt+"!important";'
    '  });'
    '}'
    'new MutationObserver(_dlStyle).observe(document.documentElement,{childList:true,subtree:true,attributes:true,attributeFilter:["class"]});'
    '</script>'
    '</body></html>'
)


def field(label_text, help_text_str, component):
    return html.Div([
        html.Label(label_text, className="field-label"),
        component,
        html.P(help_text_str, className="help-text"),
    ])


INPUT_STYLE = {
    "width": "100%", "padding": "6px 10px", "borderRadius": "6px",
    "border": "1px solid var(--input-border)", "background": "var(--input-bg)",
    "color": "var(--text)",
}
SMALL_INPUT = {
    "width": "100%", "padding": "5px 8px", "borderRadius": "6px",
    "border": "1px solid var(--input-border)", "background": "var(--input-bg)",
    "color": "var(--text)", "fontSize": "13px",
}
HEX_INPUT = {
    "width": "100%", "padding": "5px 8px", "borderRadius": "6px",
    "border": "1px solid var(--input-border)", "background": "var(--input-bg)",
    "color": "var(--text)", "fontSize": "13px", "fontFamily": "monospace",
}

# =============================================================================
# Layout
# =============================================================================

app.layout = html.Div(id="app-container", className="app-container light", children=[
    dcc.Store(id="theme-store", data="light"),
    dcc.Store(id="stored-data"),
    dcc.Store(id="col-types", data={}),
    # Top bar
    html.Div(className="top-bar", children=[
        html.Div([
            html.H1("DataLens"),
            html.P("Datei laden, Spalten auswaehlen, Diagramm erzeugen."),
        ]),
        html.Button(id="theme-toggle", className="theme-btn", n_clicks=0, children="Dark Mode"),
    ]),

    # Upload
    html.Div(className="card", children=[
        dcc.Upload(id="file-upload", multiple=False, children=html.Div(className="upload-zone", children=[
            html.Div("^", className="upload-icon"),
            html.Div("CSV oder Excel-Datei hierher ziehen", className="upload-main"),
            html.Div("oder klicken zum Auswaehlen", className="upload-sub"),
        ])),
        html.Div(id="upload-status", style={"marginTop": "12px"}),
    ]),

    # Data preview
    html.Div(id="preview-container", style={"display": "none"}, children=[
        html.Div(className="card", children=[
            html.Details(open=True, children=[
                html.Summary("Datenvorschau", className="section-title"),
                html.Div(id="col-info"),
                dash_table.DataTable(
                    id="preview-table",
                    style_table={"overflowX": "auto", "marginTop": "12px"},
                    style_cell={"textAlign": "left", "padding": "6px 10px", "fontSize": "12px",
                                "fontFamily": "Inter, sans-serif"},
                    style_header={"fontWeight": "600"}, page_size=5,
                ),
            ]),
        ]),
    ]),

    # Settings row 1
    html.Div(className="two-col", children=[
        html.Div(className="card", children=[
            html.Details(open=True, children=[
                html.Summary("Datenauswahl", className="section-title"),
                field("Zeilen gruppieren nach:",
                      "Welche Spalten als Zeilenueberschriften dienen.",
                      dcc.Dropdown(id="fixed-cols-dropdown", value=[], multi=True,
                                   placeholder="z.B. Abteilung, Region ...")),
                field("Spalten zum Vergleichen:",
                      "Welche Spalten aufgeschluesselt werden.",
                      dcc.Dropdown(id="value-cols-dropdown", value=[], multi=True,
                                   placeholder="z.B. Umsatz, Bewertung ...")),
                field("Werte ausschliessen:",
                      "Bestimmte Werte aus der Analyse entfernen.",
                      dcc.Dropdown(id="filter-dropdown", value=None, multi=True, placeholder="Optional")),
                field("Aggregation:",
                      "Wird bei Boxplot und Scatter ignoriert.",
                      dcc.Dropdown(id="aggfunc-dropdown", options=[
                          {"label": "Anzahl (Count)", "value": "count"},
                          {"label": "Summe", "value": "sum"},
                          {"label": "Durchschnitt", "value": "mean"},
                          {"label": "Prozent (100% gestapelt)", "value": "percentage"},
                      ], value="count", clearable=False)),
            ]),
        ]),
        html.Div(className="card", children=[
            html.Details(open=True, children=[
                html.Summary("Diagramm", className="section-title"),
                field("Diagrammtyp:",
                      "Achsen und Farben werden automatisch vorgeschlagen.",
                      dcc.Dropdown(id="plot-type-dropdown", options=[
                          {"label": "Balkendiagramm", "value": "bar"},
                          {"label": "Heatmap", "value": "heatmap"},
                          {"label": "Kreisdiagramm", "value": "pie"},
                          {"label": "Boxplot", "value": "box"},
                          {"label": "Streudiagramm (Scatter)", "value": "scatter"},
                      ], value="bar", clearable=False)),
                field("X-Achse:", "Horizontale Dimension.",
                      dcc.Dropdown(id="x-axis-dropdown")),
                field("Y-Achse:", "Vertikale Dimension.",
                      dcc.Dropdown(id="y-axis-dropdown")),
                field("Farbgebung:", "Spalte fuer die Farbzuordnung.",
                      dcc.Dropdown(id="color-dropdown", placeholder="Optional")),
            ]),
        ]),
    ]),

    # Boxplot options
    html.Div(id="box-options-container", style={"display": "none"}, children=[
        html.Div(className="card", children=[
            html.Details(open=True, children=[
                html.Summary("Boxplot-Optionen", className="section-title"),
                html.Div(className="two-col", style={"gap": "16px"}, children=[
                    html.Div(style={"flex": "1"}, children=[
                        field("Skalierung:",
                              "Separate Skalen bei unterschiedlichen Wertebereichen.",
                              dcc.Dropdown(id="box-scale-mode", options=[
                                  {"label": "Eigene Skala pro Spalte", "value": "facet"},
                                  {"label": "Gemeinsame Skala", "value": "shared"},
                              ], value="facet", clearable=False)),
                        field("Einzelpunkte:",
                              "Datenpunkte neben den Boxen.",
                              dcc.Dropdown(id="box-points", options=[
                                  {"label": "Keine", "value": "none"},
                                  {"label": "Nur Ausreisser", "value": "outliers"},
                                  {"label": "Alle Punkte", "value": "all"},
                              ], value="outliers", clearable=False)),
                    ]),
                    html.Div(style={"flex": "1"}, children=[
                        field("Mittelwert:",
                              "Mittelwert-Linie in der Box.",
                              dcc.Dropdown(id="box-mean", options=[
                                  {"label": "Nicht anzeigen", "value": "none"},
                                  {"label": "Mittelwert-Linie", "value": "line"},
                              ], value="line", clearable=False)),
                        field("Notched:",
                              "95%-Konfidenzintervall am Median.",
                              dcc.Dropdown(id="box-notched", options=[
                                  {"label": "Aus", "value": "off"},
                                  {"label": "An", "value": "on"},
                              ], value="off", clearable=False)),
                    ]),
                ]),
            ]),
        ]),
    ]),

    # Settings row 2: Design + Sortierung
    html.Div(className="two-col", children=[
        html.Div(className="card", children=[
            html.Details(open=False, children=[
                html.Summary("Beschriftung & Design", className="section-title"),
                field("Titel:", "Ueber dem Diagramm und als Dateiname beim Export.",
                      dcc.Input(id="title-input", type="text", placeholder="Diagrammtitel", style=INPUT_STYLE)),
                field("X-Achsen-Beschriftung:", "",
                      dcc.Input(id="xaxis-input", type="text", placeholder="automatisch", style=INPUT_STYLE)),
                field("Y-Achsen-Beschriftung:", "",
                      dcc.Input(id="yaxis-input", type="text", placeholder="automatisch", style=INPUT_STYLE)),
                html.Hr(style={"border": "none", "borderTop": "1px solid var(--border)", "margin": "16px 0"}),
                html.Div(className="inline-fields", children=[
                    html.Div([
                        html.Label("Schriftgroesse", className="field-label"),
                        dcc.Input(id="design-fontsize", type="number", value=13, min=8, max=24, step=1,
                                  style=SMALL_INPUT),
                    ]),
                    html.Div([
                        html.Label("Schriftfarbe", className="field-label"),
                        dcc.Input(id="design-fontcolor", type="text", value="#1e293b",
                                  placeholder="#1e293b", style=HEX_INPUT),
                    ]),
                    html.Div([
                        html.Label("Hintergrund", className="field-label"),
                        dcc.Input(id="design-bgcolor", type="text", value="#ffffff",
                                  placeholder="#ffffff", style=HEX_INPUT),
                    ]),
                ]),
                html.P("Hex-Farbcodes, z.B. #1e293b", className="help-text"),
                html.Div(className="inline-fields", children=[
                    html.Div([
                        html.Label("Gitternetz", className="field-label"),
                        dcc.Dropdown(id="design-grid", options=[
                            {"label": "Ja", "value": "yes"},
                            {"label": "Nein", "value": "no"},
                        ], value="yes", clearable=False),
                    ]),
                    html.Div([
                        html.Label("Farbpalette", className="field-label"),
                        dcc.Dropdown(id="design-palette", options=[
                            {"label": v["label"], "value": k} for k, v in PALETTES.items()
                        ], value="standard", clearable=False),
                    ]),
                ]),
                html.Div(id="palette-preview", className="palette-preview"),
            ]),
        ]),
        html.Div(className="card", children=[
            html.Details(open=False, children=[
                html.Summary("Sortierung", className="section-title"),
                field("X-Achse sortieren:", "'Aehnlichkeit' gruppiert Werte mit aehnlicher Verteilung.",
                      dcc.Dropdown(id="sort-x-dropdown", options=[
                          {"label": "Alphabetisch", "value": "alphabetical"},
                          {"label": "Aufsteigend (Summe)", "value": "sum_asc"},
                          {"label": "Absteigend (Summe)", "value": "sum_desc"},
                          {"label": "Aehnlichkeit (Clustering)", "value": "similarity"},
                      ], value="alphabetical", clearable=False)),
                field("Y-Achse sortieren:", "Relevant fuer Heatmaps.",
                      dcc.Dropdown(id="sort-y-dropdown", options=[
                          {"label": "Alphabetisch", "value": "alphabetical"},
                          {"label": "Aufsteigend (Summe)", "value": "sum_asc"},
                          {"label": "Absteigend (Summe)", "value": "sum_desc"},
                          {"label": "Aehnlichkeit (Clustering)", "value": "similarity"},
                      ], value="alphabetical", clearable=False)),
            ]),
        ]),
    ]),

    # Chart
    html.Div(className="card", children=[
        html.Div(id="chart-warning"),
        dcc.Graph(id="main-graph", style={"height": "70vh"}),
    ]),

    # Table
    html.Div(className="card", children=[
        html.Details(open=False, children=[
            html.Summary("Aggregierte Datentabelle", className="section-title"),
            dash_table.DataTable(
                id="result-table",
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "padding": "8px 12px", "fontSize": "13px",
                            "fontFamily": "Inter, sans-serif"},
                style_header={"fontWeight": "600"}, page_size=25,
            ),
        ]),
    ]),
])


# =============================================================================
# Callbacks
# =============================================================================

# ── Theme toggle ─────────────────────────────────────────────────────────────
@app.callback(
    Output("app-container", "className"),
    Output("theme-store", "data"),
    Output("theme-toggle", "children"),
    Output("design-fontcolor", "value"),
    Output("design-bgcolor", "value"),
    Input("theme-toggle", "n_clicks"),
    State("theme-store", "data"),
)
def toggle_theme(n, current):
    if not n:
        return "app-container light", "light", "Dark Mode", "#1e293b", "#ffffff"
    new = "dark" if current == "light" else "light"
    label = "Light Mode" if new == "dark" else "Dark Mode"
    fc = "#e2e8f0" if new == "dark" else "#1e293b"
    bg = "#151d2e" if new == "dark" else "#ffffff"
    return f"app-container {new}", new, label, fc, bg


# ── Show/hide boxplot options ────────────────────────────────────────────────
@app.callback(Output("box-options-container", "style"), Input("plot-type-dropdown", "value"))
def toggle_box_options(pt):
    return {"display": "block"} if pt == "box" else {"display": "none"}


# ── Palette preview ──────────────────────────────────────────────────────────
@app.callback(Output("palette-preview", "children"), Input("design-palette", "value"))
def update_palette_preview(pal_name):
    pal = PALETTES.get(pal_name, PALETTES["standard"])
    colors = pal["colors"] if pal["colors"] else generate_distinct_colors(10)
    return [html.Div(style={"backgroundColor": c}, className="palette-swatch") for c in colors[:15]]


# ── File upload ──────────────────────────────────────────────────────────────
@app.callback(
    Output("stored-data", "data"),
    Output("col-types", "data"),
    Output("upload-status", "children"),
    Output("preview-container", "style"),
    Output("preview-table", "data"),
    Output("preview-table", "columns"),
    Output("col-info", "children"),
    Output("fixed-cols-dropdown", "options"),
    Output("value-cols-dropdown", "options"),
    Output("filter-dropdown", "options"),
    Input("file-upload", "contents"),
    State("file-upload", "filename"),
)
def handle_upload(contents, filename):
    if contents is None:
        raise PreventUpdate
    df = parse_upload(contents, filename)
    if df is None:
        return (no_update, no_update,
                html.Span("Datei konnte nicht gelesen werden.", style={"color": "#dc2626"}),
                {"display": "none"}, [], [], [], [], [], [])
    col_types = classify_columns(df)
    cat_n = sum(1 for t in col_types.values() if t == "categorical")
    num_n = sum(1 for t in col_types.values() if t == "numeric")
    status = html.Span(
        f"'{filename}' -- {len(df)} Zeilen, {len(df.columns)} Spalten ({cat_n} kat., {num_n} num.)",
        style={"color": "#16a34a", "fontSize": "13px"},
    )
    tags = [html.Span(
        f"{c} ({'NUM' if col_types[c] == 'numeric' else 'KAT'}, {df[c].nunique()})",
        className=f"col-tag {'num' if col_types[c] == 'numeric' else 'cat'}",
    ) for c in df.columns]
    col_opts = [{"label": c, "value": c} for c in df.columns]
    str_vals = df.select_dtypes(include="object").stack().unique()
    filter_opts = [{"label": v, "value": v} for v in sorted(str_vals)]
    return (
        df.to_json(date_format="iso", orient="split"), col_types, status,
        {"display": "block"},
        df.head(5).to_dict("records"), [{"name": c, "id": c} for c in df.columns],
        html.Div(tags, style={"marginBottom": "8px"}),
        col_opts, col_opts, filter_opts,
    )


# ── Auto-configure axes ─────────────────────────────────────────────────────
@app.callback(
    Output("x-axis-dropdown", "options"), Output("x-axis-dropdown", "value"),
    Output("y-axis-dropdown", "options"), Output("y-axis-dropdown", "value"),
    Output("color-dropdown", "options"),  Output("color-dropdown", "value"),
    Input("plot-type-dropdown", "value"),
    Input("stored-data", "data"), Input("col-types", "data"),
    Input("value-cols-dropdown", "value"), Input("fixed-cols-dropdown", "value"),
)
def update_axis_options(plot_type, json_data, col_types, value_cols, fixed_cols):
    if not json_data:
        return [], None, [], None, [], None
    col_types = col_types or {}
    value_cols = value_cols or []
    fixed_cols = fixed_cols or []
    all_cols = list(col_types.keys())
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]
    num_cols = [c for c, t in col_types.items() if t == "numeric"]
    v_name  = {"label": VIRTUAL_LABELS[COL_NAME],  "value": COL_NAME}
    v_value = {"label": VIRTUAL_LABELS[COL_VALUE], "value": COL_VALUE}
    v_count = {"label": VIRTUAL_LABELS[COL_COUNT], "value": COL_COUNT}
    real_opts = [{"label": c, "value": c} for c in all_cols]

    if plot_type == "scatter":
        opts = [{"label": f"{c}  (num.)" if c in num_cols else c, "value": c} for c in all_cols]
        x = num_cols[0] if num_cols else (all_cols[0] if all_cols else None)
        y = num_cols[1] if len(num_cols) > 1 else None
        cv = cat_cols[0] if cat_cols else None
        return opts, x, opts, y, [{"label": c, "value": c} for c in all_cols], cv

    if plot_type == "box":
        xv = fixed_cols[0] if fixed_cols else (cat_cols[0] if cat_cols else COL_NAME)
        cv = xv
        co = [v_name] + [{"label": c, "value": c} for c in cat_cols]
        return [v_name] + real_opts, xv, [v_value] + real_opts, COL_VALUE, co, cv

    melted_opts = [v_name, v_value, v_count] + real_opts
    color_opts  = [v_name, v_value] + real_opts
    if plot_type == "bar":
        return melted_opts, COL_NAME, melted_opts, COL_VALUE, color_opts, COL_VALUE
    if plot_type == "heatmap":
        return melted_opts, COL_NAME, melted_opts, COL_VALUE, color_opts, None
    if plot_type == "pie":
        return melted_opts, COL_VALUE, melted_opts, COL_COUNT, color_opts, COL_VALUE
    return melted_opts, COL_NAME, melted_opts, COL_VALUE, color_opts, COL_VALUE


# ── Main chart ───────────────────────────────────────────────────────────────
@app.callback(
    Output("main-graph", "figure"), Output("main-graph", "config"),
    Output("chart-warning", "children"), Output("chart-warning", "style"),
    Output("result-table", "data"), Output("result-table", "columns"),
    Input("stored-data", "data"), Input("col-types", "data"),
    Input("fixed-cols-dropdown", "value"), Input("value-cols-dropdown", "value"),
    Input("aggfunc-dropdown", "value"), Input("plot-type-dropdown", "value"),
    Input("x-axis-dropdown", "value"), Input("y-axis-dropdown", "value"),
    Input("color-dropdown", "value"), Input("filter-dropdown", "value"),
    Input("title-input", "value"), Input("xaxis-input", "value"),
    Input("yaxis-input", "value"), Input("sort-x-dropdown", "value"),
    Input("sort-y-dropdown", "value"), Input("theme-store", "data"),
    Input("box-scale-mode", "value"), Input("box-points", "value"),
    Input("box-mean", "value"), Input("box-notched", "value"),
    Input("design-fontsize", "value"), Input("design-fontcolor", "value"),
    Input("design-bgcolor", "value"), Input("design-grid", "value"),
    Input("design-palette", "value"),
)
def update_chart(
    json_data, col_types,
    fixed_cols, value_cols, aggfunc, plot_type,
    x_axis, y_axis, color_col, filter_values,
    title_input, xaxis_input, yaxis_input, sort_x, sort_y, theme,
    box_scale, box_points, box_mean, box_notched,
    d_fs, d_fc, d_bg, d_grid, d_palette,
):
    cfg = {"toImageButtonOptions": {"format": "svg", "filename": title_input or "datalens_export",
                                     "width": 800, "height": 600, "scale": 1}}
    no_warn = ("", {"display": "none"})
    no_tbl = ([], [])
    col_types = col_types or {}
    pal = d_palette or "standard"

    font_size = max(8, min(24, d_fs or 13))
    font_color = d_fc if d_fc and _HEX_RE.match(d_fc) else None
    chart_bg = d_bg if d_bg and _HEX_RE.match(d_bg) else None
    grid_on = (d_grid != "no")

    def do_layout(fig, xlab, ylab):
        return themed_layout(fig, title_input, xlab, ylab, theme,
                             font_size=font_size, font_color=font_color,
                             chart_bg=chart_bg, grid_visible=grid_on)

    if not json_data:
        return empty_figure("Lade eine Datei, um zu starten.", theme), cfg, *no_warn, *no_tbl

    df = pd.read_json(io.StringIO(json_data), orient="split")
    fixed_cols = fixed_cols or []
    value_cols = value_cols or []

    # ── SCATTER ──────────────────────────────────────────────────────────
    if plot_type == "scatter":
        if not x_axis or not y_axis or x_axis not in df.columns or y_axis not in df.columns:
            return empty_figure("Scatter: Waehle zwei Spalten fuer X und Y.", theme), cfg, *no_warn, *no_tbl
        sdf = df.copy()
        sdf[x_axis] = pd.to_numeric(sdf[x_axis], errors="coerce")
        sdf[y_axis] = pd.to_numeric(sdf[y_axis], errors="coerce")
        sdf = sdf.dropna(subset=[x_axis, y_axis])
        if sdf.empty:
            return empty_figure("Scatter: X und Y muessen numerisch sein.", theme), cfg, *no_warn, *no_tbl
        color = color_col if color_col and color_col in sdf.columns else None
        cmap = create_color_mapping(sdf[color].tolist(), pal) if color else None
        fig = px.scatter(sdf, x=x_axis, y=y_axis, color=color, color_discrete_map=cmap, opacity=0.75)
        fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="white")))
        fig = do_layout(fig, xaxis_input or x_axis, yaxis_input or y_axis)
        return fig, cfg, *no_warn, sdf.to_dict("records"), [{"name": c, "id": c} for c in sdf.columns]

    if not value_cols:
        return empty_figure("Waehle mindestens eine Spalte unter 'Spalten zum Vergleichen'.", theme), cfg, *no_warn, *no_tbl

    melted = safe_melt(df, id_vars=fixed_cols, value_vars=value_cols)
    if melted.empty:
        return empty_figure("Keine Daten nach Umwandlung.", theme), cfg, *no_warn, *no_tbl

    if filter_values:
        melted = melted[~melted[COL_VALUE].isin(filter_values)]
        if melted.empty:
            return empty_figure("Alle Daten herausgefiltert.", theme), cfg, *no_warn, *no_tbl

    # ── BOXPLOT ──────────────────────────────────────────────────────────
    if plot_type == "box":
        m2 = melted.copy()
        m2["_num"] = pd.to_numeric(m2[COL_VALUE], errors="coerce")
        box_df = m2.dropna(subset=["_num"])
        if box_df.empty:
            return empty_figure("Boxplot: Spalten muessen numerisch sein.", theme), cfg, *no_warn, *no_tbl
        box_x = x_axis if x_axis in box_df.columns else COL_NAME
        color = color_col if color_col and color_col in box_df.columns else None
        cmap = create_color_mapping(box_df[color].tolist(), pal) if color else None
        pts_map = {"none": False, "outliers": "outliers", "all": "all"}
        nf = box_df[COL_NAME].nunique()
        use_facet = (box_scale == "facet") and (nf > 1)
        kw = dict(x=box_x, y="_num", color=color, color_discrete_map=cmap,
                   points=pts_map.get(box_points, "outliers"), notched=(box_notched == "on"))
        if use_facet:
            fig = px.box(box_df, **kw, facet_col=COL_NAME, facet_col_wrap=min(nf, 4))
            fig.update_yaxes(matches=None, showticklabels=True)
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], font=dict(size=font_size + 1)))
        else:
            fig = px.box(box_df, **kw)
        fig.update_traces(marker=dict(opacity=0.5, size=4), boxmean=(box_mean == "line"))
        fig = do_layout(fig, xaxis_input or box_x, yaxis_input or "Wert")
        out = box_df.drop(columns=["_num"])
        return fig, cfg, *no_warn, out.to_dict("records"), [{"name": c, "id": c} for c in out.columns]

    # ── Aggregation ──────────────────────────────────────────────────────
    grouped = aggregate_data(melted, aggfunc, x_axis, color_col, fixed_cols)
    if grouped.empty:
        return empty_figure("Keine Daten.", theme), cfg, *no_warn, *no_tbl

    warns = build_warnings(plot_type, color_col, grouped, melted, aggfunc, col_types, value_cols)
    w_children = [html.Div(w, className="chart-warning warn") for w in warns] if warns else ""
    w_style = {"display": "block"} if warns else {"display": "none"}

    order_x, order_y = [], []
    if x_axis and x_axis in grouped.columns:
        other = [y_axis] if (sort_x == "similarity" and y_axis and y_axis in grouped.columns) else []
        order_x = sort_dimension(grouped, x_axis, sort_x, other)
    if y_axis and y_axis in grouped.columns:
        other = [x_axis] if (sort_y == "similarity" and x_axis and x_axis in grouped.columns) else []
        order_y = sort_dimension(grouped, y_axis, sort_y, other)

    if color_col and color_col in grouped.columns:
        cmap = create_color_mapping(grouped[color_col].tolist(), pal)
    elif color_col and color_col in melted.columns:
        cmap = create_color_mapping(melted[color_col].tolist(), pal)
    else:
        cmap = {}

    # ── BAR ──────────────────────────────────────────────────────────────
    if plot_type == "bar":
        if x_axis and color_col and x_axis in grouped.columns and color_col in grouped.columns and x_axis != color_col:
            grouped = grouped.groupby([x_axis, color_col], as_index=False)[COL_COUNT].sum()
        elif x_axis and x_axis in grouped.columns and (not color_col or color_col not in grouped.columns):
            grouped = grouped.groupby([x_axis], as_index=False)[COL_COUNT].sum()
        cat_orders = {}
        if x_axis and order_x and x_axis in grouped.columns:
            cat_orders[x_axis] = [v for v in order_x if v in grouped[x_axis].unique()]
        grouped = grouped.copy()
        grouped["_lbl"] = grouped[COL_COUNT].apply(lambda v: fmt(v, aggfunc))
        fig = px.bar(grouped, x=x_axis, y=COL_COUNT,
                     color=color_col if color_col and color_col in grouped.columns else None,
                     color_discrete_map=cmap or None, category_orders=cat_orders, text="_lbl")
        fig.update_traces(textposition="auto", textangle=0, marker_line_width=0)
        fig.update_layout(barmode="stack" if aggfunc == "percentage" else "group", bargap=0.2)

    # ── HEATMAP ──────────────────────────────────────────────────────────
    elif plot_type == "heatmap":
        if not x_axis or not y_axis:
            return empty_figure("Heatmap: X und Y muessen gesetzt sein.", theme), cfg, w_children, w_style, *no_tbl
        pivot_df, err = safe_pivot(grouped, index_col=y_axis, columns_col=x_axis, value_col=COL_COUNT)
        if pivot_df is None:
            return empty_figure(f"Pivot-Fehler: {err}", theme), cfg, w_children, w_style, *no_tbl
        if order_y:
            pivot_df = pivot_df.reindex(index=[i for i in order_y if i in pivot_df.index])
        if order_x:
            pivot_df = pivot_df.reindex(columns=[c for c in order_x if c in pivot_df.columns])
        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values, x=list(pivot_df.columns), y=list(pivot_df.index),
            colorscale="Viridis", colorbar=dict(thickness=18, lenmode="fraction", len=0.6)))
        annots = []
        for i, ri in enumerate(pivot_df.index):
            for j, ci in enumerate(pivot_df.columns):
                annots.append(dict(x=ci, y=ri, text=fmt(pivot_df.iloc[i, j], aggfunc),
                                   showarrow=False, font=dict(color="white", size=max(font_size - 2, 9))))
        fig.update_layout(annotations=annots)
        if aggfunc == "percentage":
            fig.update_traces(zmin=0, zmax=100)

    # ── PIE ──────────────────────────────────────────────────────────────
    elif plot_type == "pie":
        pie_col = x_axis if x_axis and x_axis in grouped.columns else COL_VALUE
        fig = px.pie(grouped, names=pie_col, values=COL_COUNT, color_discrete_map=cmap or None)
        fig.update_traces(textposition="inside", textinfo="percent+label",
                          marker=dict(line=dict(color="white", width=1.5)))
    else:
        fig = go.Figure()

    fig = do_layout(fig, xaxis_input or (x_axis or ""), yaxis_input or (y_axis or ""))
    out_cols = [c for c in grouped.columns if c != "_lbl"]
    return fig, cfg, w_children, w_style, grouped[out_cols].to_dict("records"), [{"name": c, "id": c} for c in out_cols]


# =============================================================================
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
