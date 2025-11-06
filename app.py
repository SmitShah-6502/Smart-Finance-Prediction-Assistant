# app.py — Smart Finance Prediction Assistant (deploy-safe)

from __future__ import annotations
import os, sys, platform, traceback, logging, math, random, tempfile
from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------- Page + Global Config ----------------------------
st.set_page_config(page_title="AI Finance Pro", layout="wide")
logging.basicConfig(level=logging.INFO)
APP_DIR = Path(__file__).parent

# Debug toggle: open your app with ?debug=1 to see detailed info
DEBUG = st.query_params.get("debug", ["0"])[0] in ("1", "true", "True", "yes")
GROQ_API_KEY="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"
def _safe_list(dirpath: Path, max_items: int = 120):
    try:
        items = list(Path(dirpath).glob("**/*"))[:max_items]
        return [str(x.relative_to(APP_DIR)) for x in items]
    except Exception as e:
        return [f"<error listing {dirpath}: {e}>"]

def run_safely(fn):
    try:
        fn()
    except Exception as e:
        st.error("💥 The app crashed with an exception:")
        st.exception(e)
        st.code(traceback.format_exc())
        st.stop()

# --------------------------- Optional Imports (Lazy) -------------------------
def get_groq_client():
    """Return Groq client if available + key found, else None."""
    try:
        from groq import Groq
    except Exception:
        return None
    api_key = (
        st.secrets.get("GROQ_API_KEY")
        if hasattr(st, "secrets") else None
    ) or os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception:
        return None

def lazy_import_prophet():
    try:
        from prophet import Prophet
        return Prophet
    except Exception as e:
        st.warning("⚠️ Prophet not available. Add prophet==1.0.1 and compatible deps to requirements.")
        return None

def lazy_import_yahooquery():
    try:
        from yahooquery import Ticker
        return Ticker
    except Exception:
        st.warning("⚠️ yahooquery not installed. Add yahooquery to requirements.txt.")
        return None

def lazy_import_langchain():
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import FAISS
        from langchain.embeddings import HuggingFaceEmbeddings
        return PyPDFLoader, RecursiveCharacterTextSplitter, FAISS, HuggingFaceEmbeddings
    except Exception:
        return None, None, None, None

def lazy_import_gtts():
    try:
        from gtts import gTTS
        return gTTS
    except Exception:
        return None

# Stubs for optional project-local modules
try:
    from agents.stock_agent import analyze_stock_with_portfolio as _analyze_stock_with_portfolio
except Exception:
    def _analyze_stock_with_portfolio(*args, **kwargs):
        return "Portfolio analysis module not found. Please include agents/stock_agent.py or install its deps."

try:
    from agents.finance_chatbot_agent import multilingual_chatbot as _multilingual_chatbot, voice_input_to_text as _voice_input_to_text
except Exception:
    def _multilingual_chatbot(q: str) -> str:
        return "Chatbot module not found. Please include agents/finance_chatbot_agent.py or set API keys in secrets."
    def _voice_input_to_text() -> str:
        return "Speech recognition not configured."

# --------------------------- CSS / Branding ----------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(rgba(255,255,255,0.9), rgba(255,255,255,0.9)),
                url('https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?auto=format&fit=crop&w=1920&q=80');
    background-size: cover; background-attachment: fixed;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Inter, sans-serif; color: #1a2a44;
}
.main-title { font-size: 42px; font-weight: 800; color: #1a2a44; text-align: center; margin: 30px 0; text-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.section { background: rgba(255,255,255,0.95); border-radius: 12px; padding: 25px; margin: 20px auto; box-shadow: 0 6px 20px rgba(0,0,0,0.08); max-width: 1200px; }
.stButton>button { background: linear-gradient(90deg, #3b82f6, #1d4ed8); color: #fff; font-weight: 600; border-radius: 8px; padding: 12px 24px; border: none; transition: all .3s; }
.stButton>button:hover { background: linear-gradient(90deg, #1d4ed8, #1e40af); transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
.stTextInput>div>input, .stNumberInput>div>input, .stTextArea>div>textarea, .stSelectbox>div>div {
    border: 2px solid #3b82f6; border-radius: 8px; padding: 12px; background-color: #f9fafb; transition: border-color .3s;
}
.stTextInput>div>input:focus, .stNumberInput>div>input:focus, .stTextArea>div>textarea:focus, .stSelectbox>div>div:focus {
    border-color: #1d4ed8; box-shadow: 0 0 0 3px rgba(59,130,246,.2);
}
.stDataFrame, .stPyplot { border-radius: 12px; padding: 20px; background: #f9fafb; box-shadow: 0 2px 12px rgba(0,0,0,0.05); }
.highlight { background: #e0f2fe; padding: 20px; border-radius: 10px; border-left: 6px solid #3b82f6; margin: 15px 0; font-size: 16px; line-height: 1.6; }
.badge { display: inline-block; padding: 8px 16px; font-size: 14px; font-weight: 600; border-radius: 6px; color: #fff; }
.high-risk { background-color: #dc2626; } .neutral { background-color: #f59e0b; } .diversified { background-color: #16a34a; }
.stTabs [role="tablist"] { display: flex; justify-content: center; background: linear-gradient(180deg, #1a2a44, #3b82f6); padding: 12px; border-radius: 12px; margin: 20px auto; box-shadow: 0 4px 12px rgba(0,0,0,0.15); max-width: 1200px; }
.stTabs [role="tab"] { background: #fff; color: #1a2a44; padding: 12px 24px; margin: 0 8px; border-radius: 8px; font-weight: 600; font-size: 16px; transition: all .3s; border: 2px solid #3b82f6; display: flex; gap: 8px; }
.stTabs [role="tab"]:hover { background: #e0f2fe; transform: translateY(-2px); }
.stTabs [role="tab"][aria-selected="true"] { background: #3b82f6; color: #fff; border: 2px solid #1d4ed8; box-shadow: 0 4px 12px rgba(0,0,0,0.2); }
footer { visibility: visible; text-align: center; padding: 20px; font-size: 14px; color: #6b7280; background: rgba(255,255,255,0.9); border-top: 1px solid #e5e7eb; }
footer::after { content: "Powered by xAI"; }
</style>
""", unsafe_allow_html=True)

# --------------------------- Header -----------------------------------------
st.markdown("<div style='text-align:center'>", unsafe_allow_html=True)
logo_path = APP_DIR / "finance-logo.jpg"
if logo_path.exists():
    st.image(str(logo_path), width=120)
else:
    st.warning("⚠️ Logo image not found (finance-logo.jpg).")
st.markdown("<p style='font-style:italic;color:#1a2a44;font-size:18px;'>FINWISE: Finance Assistant</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# --------------------------- Debug Panel -------------------------------------
if DEBUG:
    st.info("🔍 Debug mode ON")
    st.write("**Python**:", sys.version)
    st.write("**Platform**:", platform.platform())
    try:
        import numpy, sklearn, plotly, matplotlib
        st.write({
            "numpy": numpy.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "plotly": plotly.__version__,
            "matplotlib": matplotlib.__version__,
        })
    except Exception as e:
        st.warning(f"Version check failed: {e}")
    st.write("**CWD**:", os.getcwd())
    st.write("**Files (first 100)**")
    st.code("\n".join(_safe_list(APP_DIR, 100)))

# --------------------------- Utilities ---------------------------------------
def normalize_ticker(t: str) -> str:
    t = (t or "").upper().strip()
    if not t:
        return t
    if t.endswith(('.NS', '.BO', '.L', '.T')):
        return t
    if t in {'TCS','RELIANCE','INFY','HDFCBANK','HINDUNILVR','ITC','LT','ICICIBANK','BHARTIARTL','SBIN'}:
        return f"{t}.NS"
    return t

def fetch_stock_data_yq(ticker: str, period: str = "1y") -> pd.DataFrame | None:
    Ticker = lazy_import_yahooquery()
    if Ticker is None:
        return None
    try:
        t = normalize_ticker(ticker)
        obj = Ticker(t)
        hist = obj.history(period=period)
        if hist is None or getattr(hist, "empty", True):
            return None
        df = hist.reset_index()
        if "symbol" in df.columns:
            df = df[df["symbol"] == t]
        df = df.rename(columns={
            "date": "ds", "open": "Open Price", "high": "High Price",
            "low": "Low Price", "close": "Close Price", "volume": "Volume"
        })
        df = df[["ds","Open Price","High Price","Low Price","Close Price","Volume"]]
        df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
        return df
    except Exception as e:
        logging.error(f"Error fetching stock data for {ticker}: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_vector_db():
    """Load PDFs from ./data into FAISS (if langchain deps exist)."""
    PyPDFLoader, TextSplitter, FAISS, HFEmb = lazy_import_langchain()
    if not all([PyPDFLoader, TextSplitter, FAISS, HFEmb]):
        return None
    pdf_dir = APP_DIR / "data"
    docs = []
    try:
        if pdf_dir.exists():
            for file in pdf_dir.glob("*.pdf"):
                loader = PyPDFLoader(str(file))
                docs.extend(loader.load())
        if not docs:
            return None
        splitter = TextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(docs)
        embeddings = HFEmb(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        logging.warning(f"Vector DB load skipped: {e}")
        return None

VECTOR_DB = load_vector_db()
GROQ_CLIENT = get_groq_client()
GROQ_MODEL = "llama-3.3-70b-versatile"

def rag_query(user_query: str) -> str:
    if GROQ_CLIENT is None:
        return "RAG disabled: GROQ_API_KEY not set in Streamlit secrets or environment."
    try:
        if VECTOR_DB:
            retriever = VECTOR_DB.as_retriever(search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(user_query)
            context_text = "\n".join(d.page_content for d in docs) if docs else ""
        else:
            context_text = ""
        if context_text:
            prompt = f"""Use the following context only if relevant. Otherwise answer directly.
Context:
{context_text}

Question: {user_query}
Answer in detail:"""
        else:
            prompt = user_query
        out = GROQ_CLIENT.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=500
        )
        resp = out.choices[0].message.content.strip()
        token_count = len(resp.split())
        perplexity = math.exp(-math.log(0.5) * token_count / 500) if token_count else 0.0
        return f"{resp}\n\n**Perplexity Score**: {perplexity:.2f}"
    except Exception as e:
        return f"RAG/Groq error: {e}"

# --------------------------- Pages -------------------------------------------
def stock_analysis_page():
    st.markdown("<div class='main-title'>💹 AI Stock Analysis & Forecasting</div>", unsafe_allow_html=True)
    st.markdown("<div class='section'>", unsafe_allow_html=True)

    ticker = st.text_input("📈 Enter stock ticker (e.g., AAPL, TSLA, RELIANCE.NS, TCS.NS):", placeholder="e.g., AAPL")
    predict_days = st.slider("📅 Days to Predict", 1, 30, 5, help="Select number of days for forecasting")

    if ticker:
        with st.spinner("Fetching data and generating insights..."):
            df_stock = fetch_stock_data_yq(ticker)
            if df_stock is not None and not df_stock.empty:
                # Currency fetch (best-effort)
                currency = "USD"
                Ticker = lazy_import_yahooquery()
                if Ticker:
                    try:
                        info = Ticker(normalize_ticker(ticker)).summary_detail
                        if isinstance(info, dict):
                            # info may be {symbol: {...}}
                            sym = normalize_ticker(ticker)
                            cval = info.get(sym, {}).get("currency")
                            if cval: currency = cval
                    except Exception:
                        pass

                st.markdown(f"### 📊 Stock Data Preview ({currency})")
                st.dataframe(df_stock.tail(min(len(df_stock), max(5, predict_days))))

                # Prophet forecast (lazy)
                Prophet = lazy_import_prophet()
                if Prophet:
                    try:
                        dfp = df_stock[["ds","Close Price"]].rename(columns={"Close Price":"y"})
                        model = Prophet(daily_seasonality=True)
                        model.fit(dfp)
                        future = model.make_future_dataframe(periods=predict_days)
                        forecast = model.predict(future)

                        # Plot
                        fig, ax = plt.subplots(figsize=(12,6))
                        ax.plot(df_stock["ds"], df_stock["Close Price"], label=f"Historical Close ({currency})")
                        ax.plot(forecast["ds"], forecast["yhat"], label=f"Forecast Close ({currency})")
                        ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], alpha=0.3, label="Confidence Interval")
                        ax.set_title(f"Stock Price Forecast for {ticker} ({currency})", fontsize=16, fontweight='bold')
                        ax.set_xlabel("Date"); ax.set_ylabel(f"Price ({currency})")
                        ax.legend(); ax.grid(True, linestyle='--', alpha=0.7)
                        st.markdown(f"### 📈 Forecast Plot (Close Price in {currency})")
                        st.pyplot(fig)

                        st.markdown(f"### 🧮 Forecasted Prices (Next {predict_days} Days)")
                        disp = forecast[["ds","yhat","yhat_lower","yhat_upper"]].tail(predict_days).copy()
                        disp.columns = ["Date", f"Predicted Close ({currency})", f"Lower Bound ({currency})", f"Upper Bound ({currency})"]
                        st.dataframe(disp)
                    except Exception as e:
                        st.warning(f"Prophet forecasting failed: {e}")

                # AI summary (Groq optional)
                if GROQ_CLIENT:
                    try:
                        recent_df = df_stock.tail(15).copy()
                        recent_df["ds"] = recent_df["ds"].dt.strftime('%Y-%m-%d')
                        recent_text = recent_df.to_string(index=False)
                        prompt = f"""You are a professional financial analyst. Provide a structured analysis of {ticker}.
All values are in {currency}. Use:
1. Trend Analysis
2. Volatility
3. Support & Resistance
4. Short-term Outlook
5. Risks

Recent stock data (last 15 rows):
{recent_text}
"""
                        out = GROQ_CLIENT.chat.completions.create(
                            model=GROQ_MODEL,
                            messages=[{"role":"user","content":prompt}],
                            temperature=0.4,
                            max_tokens=500
                        )
                        st.markdown(f"### 🧠 AI-Based Stock Analysis ({currency})")
                        st.markdown(f"<div class='highlight'>{out.choices[0].message.content.strip()}</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"AI analysis failed: {e}")
                else:
                    st.info("Set GROQ_API_KEY in Streamlit Secrets to enable AI analysis.")

                # RAG section (optional)
                st.markdown("### 📚 Knowledge Base Insights")
                st.markdown(f"<div class='highlight'>{rag_query(f'Detailed background of {ticker} stock.')}</div>", unsafe_allow_html=True)
                st.markdown("---")
                st.markdown("🔗 **Explore More**: Visit Value Research Online for detailed stock analysis.", unsafe_allow_html=True)
            else:
                st.error(f"No stock data available for {ticker}. Please check the ticker.")
                st.info("""**Valid Ticker Examples:**
- US: AAPL, TSLA, MSFT
- India (NSE): RELIANCE.NS, TCS.NS, INFY.NS
- India (BSE): RELIANCE.BO, TCS.BO, INFY.BO
- UK (LSE): BP.L, VOD.L
- Japan (TSE): 7203.T, 6758.T
""")
    st.markdown("</div>", unsafe_allow_html=True)

def portfolio_analysis_page():
    st.markdown("<div class='main-title'>📌 Personalized Portfolio Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        ticker_portfolio = st.text_input("📈 Enter stock ticker for portfolio analysis:", placeholder="e.g., RELIANCE.NS")
        shares = st.number_input("📦 Number of shares you currently own:", min_value=0, step=1, value=0)
    with col2:
        recession_rate = st.number_input("🌍 Current global recession rate (in %):", min_value=0.0, max_value=100.0, value=2.5, step=0.1)

    if ticker_portfolio and shares > 0:
        df = fetch_stock_data_yq(ticker_portfolio)
        if df is not None:
            report = _analyze_stock_with_portfolio(ticker_portfolio, df, shares, recession_rate)
            st.markdown("### 🧾 Portfolio Analysis Report")
            st.success("✅ Your personalized analysis based on current market conditions:")
            st.markdown(f"<div class='highlight'>{report}</div>", unsafe_allow_html=True)
        else:
            st.error("🚨 Unable to fetch portfolio data. Please check the ticker.")
    else:
        st.info("ℹ️ Please enter a valid stock ticker and number of shares > 0.")
    st.markdown("</div>", unsafe_allow_html=True)

def finance_chatbot_page():
    st.markdown("<div class='main-title'>💬 AI Finance Chatbot</div>", unsafe_allow_html=True)
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #1a2a44; font-style: italic;'>Your AI-powered financial advisor at your fingertips!</p>", unsafe_allow_html=True)

    lang_map = {"English": "en", "Hindi": "hi", "Gujarati": "gu"}
    choice = st.selectbox("🌐 Select Language for Response:", list(lang_map.keys()))
    lang_code = lang_map[choice]

    def translate_text(text, target):
        try:
            from deep_translator import GoogleTranslator
            return GoogleTranslator(source='auto', target=target).translate(text)
        except Exception as e:
            st.warning(f"⚠️ Translation failed: {e}")
            return text

    def get_answer(question, lang_code):
        if lang_code == "en":
            return _multilingual_chatbot(question)
        q_en = translate_text(question, "en")
        a_en = _multilingual_chatbot(q_en)
        return translate_text(a_en, lang_code)

    user_q = st.text_area("📝 Ask a finance-related question:", height=150, placeholder="e.g., What is a mutual fund?")
    if st.button("💡 Get Answer"):
        if user_q.strip():
            ans = get_answer(user_q, lang_code)
            st.markdown("**Answer:**")
            st.markdown(f"<div class='highlight'>{ans}</div>", unsafe_allow_html=True)
            gTTS = lazy_import_gtts()
            if gTTS:
                try:
                    tts = gTTS(text=ans, lang=lang_code)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
                        tts.save(tmp_audio.name)
                        audio_bytes = open(tmp_audio.name, "rb").read()
                    st.audio(audio_bytes, format="audio/mp3")
                    os.remove(tmp_audio.name)
                except Exception:
                    st.warning("⚠️ Text-to-speech failed.")
        else:
            st.warning("⚠️ Please enter a question.")

    if st.button("🎙️ Use Voice Input"):
        voice_text = _voice_input_to_text()
        st.write(f"🎤 You said: {voice_text}")
        if voice_text and not voice_text.lower().startswith(("sorry", "speech recognition failed")):
            ans = get_answer(voice_text, lang_code)
            st.markdown("**Answer:**")
            st.markdown(f"<div class='highlight'>{ans}</div>", unsafe_allow_html=True)
            gTTS = lazy_import_gtts()
            if gTTS:
                try:
                    tts = gTTS(text=ans, lang=lang_code)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
                        tts.save(tmp_audio.name)
                        audio_bytes = open(tmp_audio.name, "rb").read()
                    st.audio(audio_bytes, format="audio/mp3")
                    os.remove(tmp_audio.name)
                except Exception:
                    st.warning("⚠️ Text-to-speech failed.")
        else:
            st.warning("Voice input not available.")
    st.markdown("</div>", unsafe_allow_html=True)

def diversification_analysis_page():
    st.markdown("<div class='main-title'>🔍 Portfolio Diversification Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='section'>", unsafe_allow_html=True)

    # Load mapping from Investment.xlsx if present
    xlsx_path = APP_DIR / "Investment.xlsx"
    ticker_sector_map = {}
    if xlsx_path.exists():
        try:
            # engine=None lets pandas pick an available engine (openpyxl recommended)
            df_inv = pd.read_excel(str(xlsx_path), sheet_name="1_TOTAL", engine=None)
            df_inv = df_inv[["Name of Script", "SECTOR"]].dropna()
            ticker_sector_map = dict(zip(df_inv["Name of Script"].astype(str).str.upper(), df_inv["SECTOR"].astype(str)))
        except Exception as e:
            st.warning(f"Could not read Investment.xlsx: {e} — Provide the file or install openpyxl.")

    if not ticker_sector_map:
        st.info("ℹ️ Provide Investment.xlsx (sheet: 1_TOTAL, columns: Name of Script, SECTOR) to enable sector-based diversification.")
        # Create a minimal fallback sector map to keep the demo running
        ticker_sector_map = {"INFY":"IT","TCS":"IT","RELIANCE":"Energy","HDFCBANK":"Banking","ITC":"FMCG","LT":"Infra","SBIN":"Banking"}

    # Synthetic training
    unique_tickers = list(ticker_sector_map.keys())
    synthetic = []
    random.seed(42)
    for _ in range(300):
        k = random.randint(1, min(6, max(1, len(unique_tickers))))
        sample = random.sample(unique_tickers, k=k)
        sectors = [ticker_sector_map.get(t) for t in sample if ticker_sector_map.get(t)]
        u_cnt = len(set(sectors))
        label = "High Risk" if u_cnt < 3 else ("Neutral" if u_cnt == 3 else "Diversified Balanced")
        synthetic.append((u_cnt, label))
    train_df = pd.DataFrame(synthetic, columns=["UniqueSectorCount","DiversificationClass"])
    target_map = {"High Risk":0,"Neutral":1,"Diversified Balanced":2}
    train_df["Target"] = train_df["DiversificationClass"].map(target_map)

    # Train simple pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    X = train_df[["UniqueSectorCount"]]; y = train_df["Target"]
    pipe = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier(random_state=42))])
    pipe.fit(X, y)
    cls_decode = {v:k for k,v in target_map.items()}

    st.markdown("### 🔍 Diversification Prediction")
    user_input = st.text_input("🎯 Enter stock tickers separated by commas (e.g., UPL, INFY, RELIANCE):")
    if user_input:
        tickers = [t.strip().upper() for t in user_input.split(",") if t.strip()]
        matched_secs = [ticker_sector_map[t] for t in tickers if t in ticker_sector_map]
        u_secs = set(matched_secs)
        u_cnt = len(u_secs)
        pred = int(pipe.predict([[u_cnt]])[0])
        div_class = cls_decode[pred]
        badge = {"High Risk":"high-risk","Neutral":"neutral","Diversified Balanced":"diversified"}[div_class]
        st.markdown(f"<h3>🧠 Classification: <span class='badge {badge}'>{div_class}</span></h3>", unsafe_allow_html=True)
        summary = (f"Your portfolio includes stocks from **{u_cnt} unique sectors**: {', '.join(sorted(u_secs)) or 'N/A'}.\n\n"
                   f"🔍 Based on this, your portfolio is classified as **{div_class}**.\n\n"
                   f"📈 Diversification across multiple sectors helps reduce risk and improve stability.")
        st.markdown("### 📋 AI Summary")
        st.markdown(f"<div class='highlight'>{summary}</div>", unsafe_allow_html=True)
        st.markdown("### 📊 Sector Breakdown")
        st.dataframe(pd.DataFrame({"Ticker":[t for t in tickers if t in ticker_sector_map],
                                   "Sector":[ticker_sector_map[t] for t in tickers if t in ticker_sector_map]}))
    else:
        st.warning("🚨 Please enter stock tickers to get diversification prediction.")
    st.markdown("</div>", unsafe_allow_html=True)

def budget_recommendation_page():
    st.markdown("<div class='main-title'>💰 AI Budget Recommender</div>", unsafe_allow_html=True)
    st.markdown("<div class='section'>", unsafe_allow_html=True)

    monthly_income = st.number_input("💸 Enter your total monthly income (in your currency):", min_value=0.0, format="%.2f")
    goals = st.multiselect(
        "🎯 Select your financial goals:",
        options=["Retirement Savings", "Emergency Fund", "Debt Repayment", "Vacation Fund", "Home Purchase", "Education Savings", "Investment Growth", "General Savings"]
    )
    debt_amount = 0.0
    if "Debt Repayment" in goals:
        debt_amount = st.number_input("💳 Enter your monthly debt repayment amount (optional):", min_value=0.0, format="%.2f")

    if st.button("📊 Generate Budget Recommendation"):
        if monthly_income <= 0:
            st.warning("🚨 Please enter a valid monthly income greater than 0.")
        else:
            essentials_pct, savings_pct, discretionary_pct, debt_pct = 50.0, 30.0, 20.0, 0.0
            if "Debt Repayment" in goals and debt_amount > 0:
                debt_pct = (debt_amount / monthly_income) * 100
                discretionary_pct = max(5.0, discretionary_pct - debt_pct)
                savings_pct = 100 - essentials_pct - debt_pct - discretionary_pct
            if "Retirement Savings" in goals:
                savings_pct += 10; discretionary_pct -= 10
            if "Emergency Fund" in goals:
                savings_pct += 5; discretionary_pct -= 5
            if discretionary_pct < 5:
                discretionary_pct = 5
                savings_pct = 100 - essentials_pct - debt_pct - discretionary_pct

            st.markdown("### 📋 Budget Allocation")
            st.markdown(f"<div class='highlight'>"
                        f"- **Essentials**: **{essentials_pct:.1f}%**<br>"
                        f"- **Savings**: **{savings_pct:.1f}%**<br>"
                        f"{f'- **Debt Repayment**: **{debt_pct:.1f}%**<br>' if debt_pct>0 else ''}"
                        f"- **Discretionary**: **{discretionary_pct:.1f}%**"
                        f"</div>", unsafe_allow_html=True)

            st.markdown("### 💵 Monthly Amounts")
            st.markdown(f"<div class='highlight'>"
                        f"- Essentials: {monthly_income * essentials_pct / 100:.2f}<br>"
                        f"- Savings: {monthly_income * savings_pct / 100:.2f}<br>"
                        f"{f'- Debt Repayment: {monthly_income * debt_pct / 100:.2f}<br>' if debt_pct>0 else ''}"
                        f"- Discretionary: {monthly_income * discretionary_pct / 100:.2f}"
                        f"</div>", unsafe_allow_html=True)
            st.info("ℹ️ This is a basic recommendation. For personalized advice, consult a financial advisor.")
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------- Main Layout -------------------------------------
def main():
    tab_names = [
        "📈 Stock Analysis & Forecast",
        "📊 Portfolio Analysis",
        "💬 Finance Chatbot",
        "🔍 Diversification Analysis",
        "💰 Budget Recommendation"
    ]
    tabs = st.tabs(tab_names)
    with tabs[0]: stock_analysis_page()
    with tabs[1]: portfolio_analysis_page()
    with tabs[2]: finance_chatbot_page()
    with tabs[3]: diversification_analysis_page()
    with tabs[4]: budget_recommendation_page()

if __name__ == "__main__":
    run_safely(main)
