import csv
import gradio as gr
import joblib
import json
import os
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from get_price_data import fetch_latest_data, calculate_technical_indicators

_here = os.path.dirname(os.path.abspath(__file__))
MODEL_5D_PATH = os.path.join(_here, 'stock_model_5d.joblib')
MODEL_14D_PATH = os.path.join(_here, 'stock_model_14d.joblib')
WATCHLIST_PATH = os.path.join(_here, 'watchlist.json')


def _load_model(path):
    try:
        d = joblib.load(path)
        return d['model'], d['features']
    except Exception as e:
        print(f"Error loading model {path}: {e}")
        return None, []


model_5d, feature_names_5d = _load_model(MODEL_5D_PATH)
model_14d, feature_names_14d = _load_model(MODEL_14D_PATH)

SIGNAL_MAP = {
    1: "UP > 3%",
    0: "STABLE",
    -1: "DOWN > 3%",
}

DEFAULT_WATCHLIST = [
    # Singapore
    "U96.SI", "HLPD.SI", "H18.SI", "OV8.SI", "EB5.SI", "D05.SI",
    # Hong Kong
    "1698.HK", "1882.HK", "1997.HK", "2196.HK", "2313.HK",
    "2331.HK", "3606.HK", "6110.HK", "6169.HK", "6178.HK",
    "6690.HK", "6936.HK", "9899.HK", "9987.HK", "9988.HK",
    "9999.HK", "1398.HK", "2525.HK", "0151.HK", "0168.HK",
    "0382.HK", "0700.HK", "1513.HK", "2176.HK", "2669.HK",
    "3660.HK", "3888.HK", "1171.HK", "1179.HK", "1969.HK",
    "2648.HK", "2660.HK", "6826.HK", "9618.HK", "9979.HK",
    "9922.HK", "3686.HK", "1876.HK", "1760.HK", "0839.HK",
    "3088.HK", "2801.HK", "2839.HK", "3403.HK", "3188.HK",
    "2822.HK",
    # China A-shares
    "002304.SZ", "000568.SZ", "000858.SZ", "600809.SS",
]


_HK5_RE = re.compile(r'^0(\d{4})(\.HK)$', re.IGNORECASE)


def _normalize_ticker(raw: str) -> str:
    """Upper-case and strip leading zero from 5-digit HK codes.

    yfinance uses 4-digit HK codes: '01698.HK' → '1698.HK',
    '09999.HK' → '9999.HK', while '0700.HK' (already 4-digit) is unchanged.
    """
    t = raw.strip().upper()
    m = _HK5_RE.match(t)
    return m.group(1) + m.group(2).upper() if m else t


def _load_watchlist() -> list:
    if os.path.exists(WATCHLIST_PATH):
        try:
            with open(WATCHLIST_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT_WATCHLIST.copy()


def _save_watchlist(watchlist: list) -> None:
    try:
        with open(WATCHLIST_PATH, 'w') as f:
            json.dump(watchlist, f)
    except Exception as e:
        print(f"Failed to save watchlist: {e}")


# ── Pure inference helpers (kept for test compatibility) ─────────────────────

def build_prediction_response(ticker: str, mdl, features_array: np.ndarray, raw_data) -> dict:
    prediction = int(mdl.predict(features_array)[0])
    probabilities = mdl.predict_proba(features_array)[0].tolist()
    classes = mdl.classes_.tolist()
    prob_dict = {str(int(c)): round(p, 4) for c, p in zip(classes, probabilities)}

    confidence_up_3pct = round(prob_dict.get('1', 0.0), 4)
    confidence_down_3pct = round(prob_dict.get('-1', 0.0), 4)
    p_down = confidence_down_3pct
    edge_ratio = round(confidence_up_3pct / p_down, 2) if p_down > 0 else 99.0

    return {
        "ticker": ticker,
        "prediction": prediction,
        "signal": SIGNAL_MAP.get(prediction, "UNKNOWN"),
        "confidence_up_3pct": confidence_up_3pct,
        "confidence_down_3pct": confidence_down_3pct,
        "edge_ratio": edge_ratio,
        "probabilities": prob_dict,
        "current_price": float(raw_data['Adj_Close'].iloc[-1]),
        "last_updated": str(raw_data.index[-1].date()),
    }


def rank_scan_results(results: list, n: int = 5) -> list:
    return sorted(results, key=lambda r: (r["confidence_up_3pct"], r["edge_ratio"]), reverse=True)[:n]


# ── Data + prediction pipeline ───────────────────────────────────────────────

def _build_prediction(ticker: str, mdl, feat_names: list) -> dict | None:
    try:
        data = fetch_latest_data(ticker)
        if data is None:
            return None
        processed = calculate_technical_indicators(data)
        if processed.empty:
            return None
        latest = processed[feat_names].iloc[-1:].values
        return build_prediction_response(ticker, mdl, latest, data)
    except Exception as e:
        print(f"Prediction error for {ticker}: {e}")
        return None


def _build_dual_prediction(ticker: str) -> dict | None:
    try:
        data = fetch_latest_data(ticker)
        if data is None:
            return None
        processed = calculate_technical_indicators(data)
        if processed.empty:
            return None

        result = {
            "ticker": ticker,
            "current_price": float(data['Adj_Close'].iloc[-1]),
            "last_updated": str(data.index[-1].date()),
        }

        if model_5d is not None:
            latest = processed[feature_names_5d].iloc[-1:].values
            r5 = build_prediction_response(ticker, model_5d, latest, data)
            result.update({
                "signal_5d": r5["signal"],
                "confidence_up_5d": r5["confidence_up_3pct"],
                "confidence_down_5d": r5["confidence_down_3pct"],
                "edge_ratio_5d": r5["edge_ratio"],
                "prediction_5d": r5["prediction"],
            })

        if model_14d is not None:
            latest = processed[feature_names_14d].iloc[-1:].values
            r14 = build_prediction_response(ticker, model_14d, latest, data)
            result.update({
                "signal_14d": r14["signal"],
                "confidence_up_14d": r14["confidence_up_3pct"],
                "confidence_down_14d": r14["confidence_down_3pct"],
                "edge_ratio_14d": r14["edge_ratio"],
                "prediction_14d": r14["prediction"],
            })

        return result
    except Exception as e:
        print(f"Dual prediction error for {ticker}: {e}")
        return None


# ── HTML renderers ────────────────────────────────────────────────────────────

_MONO = "font-family:'Source Code Pro',monospace;"
_ERR_STYLE = f"color:#6b1616;font-size:15px;{_MONO}"



def _result_html(r: dict, label: str = "") -> str:
    up_pct = round(r['confidence_up_3pct'] * 100)
    dn_pct = round(r['confidence_down_3pct'] * 100)
    if r['prediction'] == 1:
        sig_color = "#1e4d17"
    elif r['prediction'] == -1:
        sig_color = "#6b1616"
    else:
        sig_color = "#5a4a15"
    label_html = (
        f'<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;'
        f'color:#888;margin-bottom:4px;">{label}</div>'
        if label else ""
    )
    horizon = r.get('horizon', 7)
    return f"""
<div style="border:1px solid #b8b2aa;padding:18px 20px;background:#faf8f4;{_MONO}margin:4px 0;">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;
              border-bottom:1px solid #d0cac2;padding-bottom:12px;margin-bottom:14px;">
    <div>
      {label_html}
      <div style="font-size:24px;font-weight:700;letter-spacing:1px;color:#1a1a18;">{r['ticker']}</div>
      <div style="font-size:16px;color:#555;margin-top:3px;">{r['current_price']:.3f}</div>
    </div>
    <div style="text-align:right;">
      <div style="color:{sig_color};font-weight:700;font-size:16px;letter-spacing:1px;">{r['signal']}</div>
      <div style="font-size:13px;color:#888;margin-top:3px;">EDGE {r['edge_ratio']:.2f}&times;</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;font-size:15px;">
    <span style="width:180px;color:#555;flex-shrink:0;">UP &gt;3% in {horizon} days</span>
    <div style="flex:1;height:9px;background:#e5e0d9;border:1px solid #c8c2ba;">
      <div style="width:{up_pct}%;height:100%;background:#1e4d17;"></div>
    </div>
    <span style="width:42px;text-align:right;font-weight:700;color:#1a1a18;">{up_pct}%</span>
  </div>
  <div style="display:flex;align-items:center;gap:10px;font-size:15px;">
    <span style="width:180px;color:#555;flex-shrink:0;">DOWN &gt;3% in {horizon} days</span>
    <div style="flex:1;height:9px;background:#e5e0d9;border:1px solid #c8c2ba;">
      <div style="width:{dn_pct}%;height:100%;background:#6b1616;"></div>
    </div>
    <span style="width:42px;text-align:right;font-weight:700;color:#1a1a18;">{dn_pct}%</span>
  </div>
  <div style="font-size:12px;color:#aaa;text-align:right;margin-top:12px;
              border-top:1px solid #e5e0d9;padding-top:8px;">DATA AS OF {r['last_updated']}</div>
</div>"""


def _scan_html(results: list) -> str:
    if not results:
        return f'<p style="{_ERR_STYLE}">No data available for any ticker in your watchlist.</p>'

    def _sig_color(pred):
        if pred == 1:
            return "#1e4d17"
        elif pred == -1:
            return "#6b1616"
        return "#5a4a15"

    rows = ""
    for i, r in enumerate(results):
        up_pct_5d = round(r.get("confidence_up_5d", 0) * 100)
        sig5_color = _sig_color(r.get("prediction_5d", 0))
        sig14_color = _sig_color(r.get("prediction_14d", 0))
        rows += (
            f'<tr style="border-bottom:1px solid #e5e0d9;">'
            f'<td style="padding:8px 10px;color:#aaa;">{i+1}</td>'
            f'<td style="padding:8px 10px;font-weight:700;color:#1a1a18;">{r["ticker"]}</td>'
            f'<td style="padding:8px 10px;color:#333;">{r["current_price"]:.3f}</td>'
            f'<td style="padding:8px 10px;color:{sig5_color};font-weight:700;">'
            f'{r.get("signal_5d", "N/A")}</td>'
            f'<td style="padding:8px 10px;font-weight:700;color:#1a1a18;">'
            f'{r.get("edge_ratio_5d", 0):.2f}&times;</td>'
            f'<td style="padding:8px 10px;color:{sig14_color};font-weight:700;">'
            f'{r.get("signal_14d", "N/A")}</td>'
            f'<td style="padding:8px 10px;font-weight:700;color:#1a1a18;">'
            f'{r.get("edge_ratio_14d", 0):.2f}&times;</td>'
            f'<td style="padding:8px 10px;color:#333;">{up_pct_5d}%</td>'
            f'</tr>'
        )
    th = ("padding:8px 10px;font-size:12px;text-transform:uppercase;"
          "letter-spacing:1px;text-align:left;border-bottom:2px solid #1a1a18;color:#1a1a18;")
    return (
        f'<table style="width:100%;border-collapse:collapse;font-size:14px;{_MONO}margin:4px 0;">'
        f'<thead><tr>'
        f'<th style="{th}">#</th>'
        f'<th style="{th}">TICKER</th>'
        f'<th style="{th}">PRICE</th>'
        f'<th style="{th}">SIGNAL 5D</th>'
        f'<th style="{th}">EDGE 5D</th>'
        f'<th style="{th}">SIGNAL 14D</th>'
        f'<th style="{th}">EDGE 14D</th>'
        f'<th style="{th}">UP CONF 5D</th>'
        f'</tr></thead>'
        f'<tbody>{rows}</tbody>'
        f'</table>'
    )


# ── Gradio action handlers ────────────────────────────────────────────────────

def _wl_updates(wl: list):
    return wl, gr.update(choices=wl, value=[])


def analyze(ticker: str, watchlist: list):
    ticker = _normalize_ticker(ticker)
    if not ticker:
        err = f'<p style="{_ERR_STYLE}">Enter a ticker symbol.</p>'
        return (err, "", *_wl_updates(watchlist))

    html_5d = html_14d = ""
    r5 = r14 = None

    if model_5d is not None:
        r5 = _build_prediction(ticker, model_5d, feature_names_5d)
        html_5d = (_result_html(r5, label="5-DAY FORECAST") if r5
                   else f'<p style="{_ERR_STYLE}">No data for {ticker} (5D).</p>')
    else:
        html_5d = f'<p style="{_ERR_STYLE}">5-day model not loaded.</p>'

    if model_14d is not None:
        r14 = _build_prediction(ticker, model_14d, feature_names_14d)
        html_14d = (_result_html(r14, label="14-DAY FORECAST") if r14
                    else f'<p style="{_ERR_STYLE}">No data for {ticker} (14D).</p>')
    else:
        html_14d = f'<p style="{_ERR_STYLE}">14-day model not loaded.</p>'

    if ticker not in watchlist and (r5 is not None or r14 is not None):
        watchlist = watchlist + [ticker]
        _save_watchlist(watchlist)

    return (html_5d, html_14d, *_wl_updates(watchlist))


def add_ticker(ticker: str, watchlist: list):
    ticker = _normalize_ticker(ticker)
    if ticker and ticker not in watchlist:
        watchlist = watchlist + [ticker]
        _save_watchlist(watchlist)
    return ("", *_wl_updates(watchlist))


def import_csv(file, watchlist: list):
    if file is None:
        return ("", *_wl_updates(watchlist))
    try:
        filepath = file.path if hasattr(file, 'path') else file
        header_kw = ['ticker', 'symbol', 'code', 'stock', 'instrument']
        new_tickers = []
        with open(filepath, newline='', encoding='utf-8-sig') as f:
            rows = list(csv.reader(f))
        if not rows:
            return ("", *_wl_updates(watchlist))
        # Detect header row and ticker column
        start_row, ticker_col = 0, 0
        first_row_lower = [c.strip().lower().strip('"\'') for c in rows[0]]
        for i, cell in enumerate(first_row_lower):
            if any(kw in cell for kw in header_kw):
                start_row, ticker_col = 1, i
                break
        for row in rows[start_row:]:
            if len(row) > ticker_col:
                t = _normalize_ticker(row[ticker_col].strip('"\''))
                if t and t not in watchlist and t not in new_tickers:
                    new_tickers.append(t)
        watchlist = watchlist + new_tickers
        _save_watchlist(watchlist)
    except Exception as e:
        print(f"CSV import error: {e}")
    return ("", *_wl_updates(watchlist))


def remove_tickers(to_remove: list, watchlist: list):
    if to_remove:
        watchlist = [t for t in watchlist if t not in to_remove]
        _save_watchlist(watchlist)
    return _wl_updates(watchlist)


def scan_watchlist(watchlist: list) -> str:
    if not watchlist:
        return f'<p style="{_ERR_STYLE}">Watchlist is empty.</p>'
    if model_5d is None and model_14d is None:
        return f'<p style="{_ERR_STYLE}">No models loaded.</p>'
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_build_dual_prediction, watchlist))
    valid = [r for r in results if r is not None]
    top10 = sorted(
        valid,
        key=lambda r: (r.get("confidence_up_5d", 0), r.get("edge_ratio_5d", 0)),
        reverse=True,
    )[:10]
    return _scan_html(top10)


# ── Theme ─────────────────────────────────────────────────────────────────────

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.stone,
    secondary_hue=gr.themes.colors.stone,
    neutral_hue=gr.themes.colors.stone,
    font=gr.themes.GoogleFont("Source Code Pro"),
    text_size=gr.themes.sizes.text_lg,
    radius_size=gr.themes.sizes.radius_none,
).set(
    body_background_fill="#f4f1ec",
    body_text_color="#1a1a18",
    button_primary_background_fill="#1a1a18",
    button_primary_text_color="#f4f1ec",
    button_primary_background_fill_hover="#333",
    button_primary_border_color="#1a1a18",
    button_secondary_background_fill="#eae6e0",
    button_secondary_text_color="#1a1a18",
    button_secondary_border_color="#a8a49e",
    block_background_fill="#f4f1ec",
    block_border_color="#c8c0b8",
    block_border_width="0px",
    input_background_fill="#ffffff",
    input_border_color="#b8b2aa",
    input_border_color_focus="#1a1a18",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Source+Code+Pro:ital,wght@0,400;0,500;0,600;0,700&display=swap');

/* ── Page & universal font ── */
html { background: #f4f1ec !important; }
body { background: #f4f1ec !important; color: #1a1a18 !important; }
*, *::before, *::after { font-family: 'Source Code Pro', monospace !important; }

/* ── Container ── */
.gradio-container {
    max-width: 700px !important;
    padding: 0 28px 60px !important;
    background: #f4f1ec !important;
}
footer, .show-api { display: none !important; }

/* ── Inputs ── */
input[type="text"], input[type="search"], textarea {
    color: #1a1a18 !important;
    background: #ffffff !important;
    border: 1.5px solid #b0aaa4 !important;
    font-size: 16px !important;
    padding: 0 12px !important;
}
input[type="text"]:focus, input[type="search"]:focus {
    border-color: #1a1a18 !important;
    box-shadow: none !important;
    outline: none !important;
}
input::placeholder, textarea::placeholder {
    color: #b0aaa4 !important;
    font-size: 15px !important;
    opacity: 1 !important;
}

/* ── Buttons ── */
button {
    font-size: 14px !important;
    font-weight: 600 !important;
    letter-spacing: 0.8px !important;
    transition: opacity 0.15s !important;
}
button:active { opacity: 0.75 !important; }

/* ── Block labels (Gradio auto-labels) ── */
.label-wrap span, label > span:first-child {
    font-size: 11px !important;
    color: #aaa !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
}

/* ── Section headers ── */
.sec-label {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #888;
    border-bottom: 1.5px solid #1a1a18;
    padding-bottom: 6px;
    margin-bottom: 12px;
}
.sub-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #aaa;
    margin: 14px 0 5px;
}

/* ── CheckboxGroup — watchlist ── */
#wl-rm {
    background: #ffffff !important;
    border: 1.5px solid #b0aaa4 !important;
    border-radius: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}
#wl-rm .wrap, #wl-rm .form {
    background: #ffffff !important;
    max-height: 280px !important;
    overflow-y: auto !important;
    padding: 10px 14px !important;
    display: flex !important;
    flex-wrap: wrap !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
}
#wl-rm fieldset {
    background: #ffffff !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
    display: flex !important;
    flex-wrap: wrap !important;
}
#wl-rm label {
    background: #ffffff !important;
    color: #1a1a18 !important;
    border-radius: 0 !important;
    border: none !important;
    box-shadow: none !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 6px !important;
    padding: 4px 18px 4px 4px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    pointer-events: auto !important;
    white-space: nowrap !important;
}
#wl-rm label:hover { background: #f0ede7 !important; }
#wl-rm input[type="checkbox"] {
    pointer-events: auto !important;
    cursor: pointer !important;
    accent-color: #1a1a18 !important;
    flex-shrink: 0 !important;
}
#wl-rm .label-wrap { display: none !important; }
"""

# ── UI ────────────────────────────────────────────────────────────────────────

INITIAL_WATCHLIST = _load_watchlist()

with gr.Blocks(title="AlphaPulse") as demo:
    watchlist_state = gr.State(INITIAL_WATCHLIST)

    # ── Header ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center;padding:28px 0 16px;border-bottom:2px solid #1a1a18;margin-bottom:20px;">
      <div style="font-size:26px;font-weight:700;letter-spacing:4px;color:#1a1a18;
                  font-family:'Source Code Pro',monospace;">ALPHAPULSE</div>
      <div style="font-size:10px;letter-spacing:4px;text-transform:uppercase;color:#888;
                  margin-top:5px;font-family:'Source Code Pro',monospace;">STOCK FORECAST AI</div>
    </div>
    """)

    # ── Analyze ───────────────────────────────────────────────────────────────
    gr.HTML('<div class="sec-label">── ANALYZE ──</div>')
    with gr.Row():
        ticker_in = gr.Textbox(
            placeholder="e.g. 0700.HK or AAPL",
            show_label=False,
            scale=4,
        )
        analyze_btn = gr.Button("ANALYZE", variant="primary", scale=1)
    result_out = gr.HTML(value="")

    # ── Watchlist ─────────────────────────────────────────────────────────────
    gr.HTML('<div class="sec-label" style="margin-top:18px;">── WATCHLIST ──</div>')
    remove_cg = gr.CheckboxGroup(
        choices=INITIAL_WATCHLIST,
        label="",
        value=[],
        elem_id="wl-rm",
    )
    with gr.Row():
        remove_btn = gr.Button("REMOVE SELECTED", variant="secondary", scale=1)
        gr.HTML('<div></div>', visible=True)

    # ── Add tickers ───────────────────────────────────────────────────────────
    gr.HTML('<div class="sub-label">ADD TICKERS</div>')
    with gr.Row():
        add_in = gr.Textbox(
            placeholder="e.g. 0700.HK",
            show_label=False,
            scale=3,
        )
        add_btn = gr.Button("ADD", scale=1)
        csv_btn = gr.UploadButton(
            "IMPORT CSV",
            file_types=[".csv"],
            scale=1,
        )

    # ── Scan ──────────────────────────────────────────────────────────────────
    gr.HTML('<div class="sec-label" style="margin-top:18px;">── SCAN ──</div>')
    scan_btn = gr.Button("SCAN WATCHLIST", variant="primary")
    scan_out = gr.HTML(value="")

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.HTML(
        '<p style="font-size:10px;color:#bbb;text-align:center;margin-top:20px;'
        "letter-spacing:1px;font-family:'Source Code Pro',monospace;\">"
        "NOT FINANCIAL ADVICE · BASED ON HISTORICAL PATTERNS ONLY</p>"
    )

    # ── Wiring ────────────────────────────────────────────────────────────────
    _wl_outs = [watchlist_state, remove_cg]

    analyze_btn.click(
        fn=analyze,
        inputs=[ticker_in, watchlist_state],
        outputs=[result_out] + _wl_outs,
    )
    ticker_in.submit(
        fn=analyze,
        inputs=[ticker_in, watchlist_state],
        outputs=[result_out] + _wl_outs,
    )
    add_btn.click(
        fn=add_ticker,
        inputs=[add_in, watchlist_state],
        outputs=[add_in] + _wl_outs,
    )
    add_in.submit(
        fn=add_ticker,
        inputs=[add_in, watchlist_state],
        outputs=[add_in] + _wl_outs,
    )
    csv_btn.upload(
        fn=import_csv,
        inputs=[csv_btn, watchlist_state],
        outputs=[add_in] + _wl_outs,
    )
    remove_btn.click(
        fn=remove_tickers,
        inputs=[remove_cg, watchlist_state],
        outputs=_wl_outs,
    )
    scan_btn.click(
        fn=scan_watchlist,
        inputs=[watchlist_state],
        outputs=[scan_out],
    )


if __name__ == "__main__":
    demo.launch(server_port=7860, theme=theme, css=CSS)
