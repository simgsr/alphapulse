import os

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama_local")
OLLAMA_LOCAL_MODEL = os.getenv("OLLAMA_LOCAL_MODEL", "llama3")

_SYSTEM_PROMPT = (
    "You are a financial signal interpreter. "
    "Summarise the model output in 3-5 plain-English sentences. "
    "Do not give buy/sell advice. "
    "Do not invent information not provided."
)

_INDICATOR_LABELS = {
    "RSI_14": "RSI (14-day)",
    "MACD_hist": "MACD Histogram",
    "SMA_20_ratio": "Price / SMA-20",
    "Volume_ratio_20": "Volume Ratio (20-day)",
    "ATR_ratio": "ATR Ratio",
    "Returns_5d": "5-Day Return",
    "Returns_20d": "20-Day Return",
    "BB_pct_b": "Bollinger %B",
}


def get_llm(provider: str = LLM_PROVIDER, temperature: float = 0):
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(temperature=temperature)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-pro", temperature=temperature)
    from langchain_ollama import ChatOllama
    return ChatOllama(model=OLLAMA_LOCAL_MODEL, temperature=temperature)


def explain_signal(
    ticker: str,
    price: float,
    result_5d: dict,
    result_14d: dict,
    indicators: dict,
) -> str:
    ind_lines = "\n".join(
        f"  {_INDICATOR_LABELS.get(k, k)}: {v:.4f}"
        for k, v in indicators.items()
        if k in _INDICATOR_LABELS
    )
    prompt = (
        f"Ticker: {ticker}  |  Current price: {price:.3f}\n\n"
        f"5-day forecast:\n"
        f"  Signal: {result_5d.get('signal', 'N/A')}\n"
        f"  Confidence (UP >3%): {result_5d.get('confidence_up_3pct', 0)*100:.1f}%\n"
        f"  Edge ratio: {result_5d.get('edge_ratio', 0):.2f}x\n\n"
        f"14-day forecast:\n"
        f"  Signal: {result_14d.get('signal', 'N/A')}\n"
        f"  Confidence (UP >3%): {result_14d.get('confidence_up_3pct', 0)*100:.1f}%\n"
        f"  Edge ratio: {result_14d.get('edge_ratio', 0):.2f}x\n\n"
        f"Key indicators:\n{ind_lines}"
    )
    from langchain_core.messages import HumanMessage, SystemMessage
    llm = get_llm()
    response = llm.invoke([SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=prompt)])
    return response.content


def summarize_scan(scan_rows: list) -> str:
    if not scan_rows:
        return "No scan results to summarise."
    rows_text = "\n".join(
        f"  {r['ticker']}: 5d={r.get('signal_5d','N/A')} ({r.get('confidence_up_5d',0)*100:.0f}%), "
        f"14d={r.get('signal_14d','N/A')} ({r.get('confidence_up_14d',0)*100:.0f}%)"
        for r in scan_rows
    )
    prompt = (
        f"Watchlist scan results:\n{rows_text}\n\n"
        "Summarise the overall pattern of signals across this watchlist in one paragraph."
    )
    from langchain_core.messages import HumanMessage, SystemMessage
    llm = get_llm()
    response = llm.invoke([SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=prompt)])
    return response.content
