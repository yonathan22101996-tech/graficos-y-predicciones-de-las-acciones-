import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from pathlib import Path
from html import escape

OUTPUT_FILE = Path("a:/python/proyecto03/informe_predicciones_interactivo.html")
TICKERS = {
    "Apple": "AAPL",
    "Nvidia": "NVDA",
    "Microsoft": "MSFT",
}


def build_forecast(series: pd.Series, periods: int = 180, high_margin: float = 0.08, low_margin: float = 0.08):
    recent = series.dropna().tail(min(180, len(series.dropna())))
    x = np.arange(len(recent))
    y = recent.values.astype(float)
    slope, intercept = np.polyfit(x, y, 1)

    future_index = pd.date_range(
        start=recent.index[-1] + pd.tseries.frequencies.to_offset("1D"),
        periods=periods,
        freq="B",
    )
    future_x = np.arange(len(recent), len(recent) + periods)
    pred_mean = slope * future_x + intercept
    pred_high = pred_mean * (1 + high_margin)
    pred_low = pred_mean * (1 - low_margin)

    return recent, pd.DataFrame(
        {
            "fecha": future_index,
            "valor_medio": pred_mean,
            "valor_alto": pred_high,
            "valor_minimo": pred_low,
        }
    )


def build_chart(company: str, ticker: str, hist: pd.Series, forecast: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist.index,
            y=hist.values,
            mode="lines",
            name="Histórico",
            line=dict(color="#7dd3fc", width=2),
            hovertemplate="%{x|%d %b %Y}<br>Precio: $%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["fecha"],
            y=forecast["valor_medio"],
            mode="lines",
            name="Valor medio",
            line=dict(color="#f59e0b", width=2.5, dash="dot"),
            hovertemplate="%{x|%d %b %Y}<br>Valor medio: $%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["fecha"],
            y=forecast["valor_alto"],
            mode="lines",
            name="Valor alto",
            line=dict(color="#34d399", width=2),
            hovertemplate="%{x|%d %b %Y}<br>Valor alto: $%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["fecha"],
            y=forecast["valor_minimo"],
            mode="lines",
            name="Valor mínimo",
            line=dict(color="#f43f5e", width=2),
            hovertemplate="%{x|%d %b %Y}<br>Valor mínimo: $%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(forecast["fecha"]) + list(reversed(forecast["fecha"])),
            y=list(forecast["valor_alto"]) + list(reversed(forecast["valor_minimo"])),
            fill="toself",
            fillcolor="rgba(148, 163, 184, 0.18)",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=f"{company} ({ticker}) - Predicción de precio a 6 meses",
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=20, t=70, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Fecha",
        yaxis_title="Precio (USD)",
    )
    return fig


def build_summary(company: str, hist: pd.Series, forecast: pd.DataFrame):
    last_price = round(float(hist.iloc[-1]), 2)
    last_pred_mid = round(float(forecast["valor_medio"].iloc[-1]), 2)
    last_pred_high = round(float(forecast["valor_alto"].iloc[-1]), 2)
    last_pred_low = round(float(forecast["valor_minimo"].iloc[-1]), 2)
    return f"""
    <div class=\"summary-card\">
      <h3>{escape(company)}</h3>
      <p><strong>Precio actual:</strong> ${last_price:.2f}</p>
      <p><strong>Predicción media:</strong> ${last_pred_mid:.2f}</p>
      <p><strong>Predicción alta:</strong> ${last_pred_high:.2f}</p>
      <p><strong>Predicción mínima:</strong> ${last_pred_low:.2f}</p>
    </div>
    """


def render_html(charts: list, summaries: list):
    chart_html = "\n".join(charts)
    summary_html = "\n".join(summaries)
    return f"""<!doctype html>
<html lang=\"es\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
    <title>Informe interactivo de predicciones</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 0; background: #0f172a; color: #e2e8f0; }}
      .page {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
      .hero {{ background: linear-gradient(135deg, #1d4ed8, #0f172a); padding: 28px; border-radius: 18px; margin-bottom: 24px; }}
      .hero h1 {{ margin: 0 0 8px; font-size: 2rem; }}
      .hero p {{ margin: 0; color: #cbd5e1; line-height: 1.6; }}
      .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }}
      .summary-card {{ background: #111827; padding: 16px 18px; border-radius: 14px; border: 1px solid #334155; }}
      .summary-card h3 {{ margin-top: 0; margin-bottom: 8px; color: #f8fafc; }}
      .summary-card p {{ margin: 6px 0; color: #cbd5e1; }}
      .chart-card {{ background: white; color: #0f172a; padding: 16px; border-radius: 16px; margin-bottom: 20px; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.15); }}
      .download {{ display: inline-block; margin-top: 8px; padding: 10px 14px; background: #38bdf8; color: #0f172a; font-weight: 700; border-radius: 999px; text-decoration: none; }}
      .download:hover {{ background: #7dd3fc; }}
    </style>
  </head>
  <body>
    <div class=\"page\">
      <div class=\"hero\">
        <h1>Informe interactivo de predicciones</h1>
        <p>Este reporte muestra el comportamiento histórico de Apple, Nvidia y Microsoft, junto con una proyección simple a 6 meses para un valor medio, alto y mínimo. Los gráficos son interactivos y pueden explorarse con hover, zoom y selección.</p>
        <a class=\"download\" href=\"informe_predicciones_interactivo.html\" download>Descargar este informe</a>
      </div>
      <div class=\"summary-grid\">{summary_html}</div>
      {chart_html}
    </div>
  </body>
</html>"""


def main():
    data = yf.download(list(TICKERS.values()), period="2y", auto_adjust=False, progress=False)
    close = data["Close"].copy()
    close.columns = list(TICKERS.keys())

    charts = []
    summaries = []
    for company, ticker in TICKERS.items():
        hist = close[company].dropna()
        recent, forecast = build_forecast(hist)
        fig = build_chart(company, ticker, recent, forecast)
        charts.append(f"<div class=\"chart-card\">{fig.to_html(full_html=False, include_plotlyjs='cdn')}</div>")
        summaries.append(build_summary(company, recent, forecast))

    html = render_html(charts, summaries)
    OUTPUT_FILE.write_text(html, encoding="utf-8")
    print(f"Archivo creado: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
