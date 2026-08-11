import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pandas_ta_classic as ta
import hashlib
from financial_calculator import FinancialCalculator
from data_fetcher import YahooFinanceFetcher
from llm_integrator import LLMIntegrator
from diagnostics import diagnose_financial_data, print_diagnostics
from rag_system import FinancialRAGSystem

st.set_page_config(page_title="Fundamental Research Copilot", layout="wide")
st.title("Fundamental Research Copilot")
st.caption("Evidence-based fundamental analysis using financial statements, company context, and retrieval-augmented generation.")

# NOTE: HuggingFace embeddings are automatically cached by Streamlit on first use.
# This means the first startup may take 20-30s as the model downloads (~100MB).
# Subsequent runs are instant. This works on Streamlit Cloud (free tier).

# Initialize session state
if "api_key" not in st.session_state:
    st.session_state.api_key = None
if "provider" not in st.session_state:
    st.session_state.provider = None
if "ticker_data_loaded" not in st.session_state:
    st.session_state.ticker_data_loaded = False
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "ticker" not in st.session_state:
    st.session_state.ticker = None
if "financial_metrics" not in st.session_state:
    st.session_state.financial_metrics = {}
if "llm_client" not in st.session_state:
    st.session_state.llm_client = None
if "llm_client_signature" not in st.session_state:
    st.session_state.llm_client_signature = None
if "initial_summary_generated" not in st.session_state:
    st.session_state.initial_summary_generated = False
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None

# Step 1: API Configuration
st.header("1. Configure the research model")
col1, col2 = st.columns(2)

with col1:
    provider = st.selectbox(
        "LLM provider",
        ("OpenAI", "DeepSeek", "Ollama (local)"),
        key="provider_select"
    )

with col2:
    api_key_required = provider != "Ollama (local)"
    api_key = st.text_input(
        f"API key for {provider}" if api_key_required else "API key (not required for Ollama)",
        type="password",
        disabled=not api_key_required,
        key="api_key_input",
        help="The key is kept in Streamlit session state and is not written to disk."
    )

if provider == "Ollama (local)":
    st.info("Ollama runs locally and requires a running Ollama server with the selected model.")

credentials_ready = bool(provider) and (bool(api_key) if api_key_required else True)
if credentials_ready:
    effective_api_key = api_key if api_key_required else None
    client_signature = (
        provider,
        hashlib.sha256((effective_api_key or '').encode()).hexdigest(),
    )
    st.session_state.api_key = effective_api_key
    st.session_state.provider = provider

    try:
        credentials_changed = (
            st.session_state.llm_client_signature is not None
            and st.session_state.llm_client_signature != client_signature
        )
        if st.session_state.llm_client is None or credentials_changed:
            st.session_state.llm_client = LLMIntegrator(provider, effective_api_key)
            st.session_state.llm_client_signature = client_signature
            if credentials_changed:
                st.session_state.ticker_data_loaded = False
                st.session_state.initial_summary_generated = False
                st.session_state.conversation_history = []
                st.session_state.rag_system = None
        st.success("Research model configured")
    except Exception as e:
        st.session_state.llm_client = None
        st.session_state.llm_client_signature = None
        st.error(f"Failed to initialize LLM: {str(e)}")
        st.stop()
else:
    st.warning("Provide an API key to continue, or choose Ollama for a local model.")
    st.stop()

# Step 2: Stock Ticker Input
st.header("2. Select a company")
ticker = st.text_input(
    "Enter stock ticker (e.g., ORCL, AAPL, MSFT):",
    key="ticker_input",
    placeholder="ORCL"
).upper()

if ticker and st.button("Load Financial Data", type="primary"):
    with st.spinner(f"Loading financial data for {ticker}..."):
        try:
            fetcher = YahooFinanceFetcher(ticker)
            financial_data = fetcher.extract_financial_data()
            
            if financial_data:
                st.session_state.ticker = ticker
                st.session_state.ticker_data_loaded = True
                
                # Calculate metrics
                calculator = FinancialCalculator(ticker)
                metrics = calculator.calculate_all_metrics(financial_data)
                formatted_metrics = calculator.format_metrics()
                
                # Get quarterly history for LLM context
                quarterly_history = fetcher.get_quarterly_history(num_quarters=8)
                annual_history = fetcher.get_annual_history(num_years=2)
                company_profile = fetcher.get_company_profile()
                
                st.session_state.financial_metrics = {
                    'raw': metrics,
                    'formatted': formatted_metrics,
                    'financial_data': financial_data,
                    'fetcher': fetcher,
                    'quarterly_history': quarterly_history,
                    'annual_history': annual_history,
                    'company_profile': company_profile,
                }
                
                # Build RAG system
                st.session_state.rag_system = None
                if st.session_state.llm_client:
                    try:
                        rag_system = FinancialRAGSystem(
                            st.session_state.api_key,
                            provider=st.session_state.provider
                        )
                        rag_system.build_knowledge_base(
                            ticker,
                            financial_data,
                            quarterly_history,
                            annual_history,
                            company_profile,
                        )
                        if rag_system.retriever is not None:
                            st.session_state.rag_system = rag_system
                        else:
                            st.warning("RAG index could not be built; using direct model analysis.")
                    except Exception as rag_e:
                        st.warning(f"RAG system - using fallback: {str(rag_e)}")
                
                st.success(f"Data loaded successfully for {ticker}")
            else:
                st.error(f"Could not fetch data for ticker {ticker}. Please check if it's valid.")
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")

if not st.session_state.ticker_data_loaded:
    st.info("Enter a stock ticker and click 'Load Financial Data' to continue")
    st.stop()

# FINANCIAL ANALYSIS SECTION
st.header(f"Financial Analysis for {st.session_state.ticker}")

# Display financial metrics
st.subheader("Key Financial Metrics")
col1, col2, col3, col4 = st.columns(4)

formatted_metrics = st.session_state.financial_metrics['formatted']
metric_items = list(formatted_metrics.items())

for idx, (metric_name, metric_value) in enumerate(metric_items):
    with [col1, col2, col3, col4][idx % 4]:
        st.metric(label=metric_name, value=metric_value)

st.divider()

# Data diagnostics (expandable)
with st.expander("Data Diagnostics"):
    diagnostics = diagnose_financial_data(st.session_state.financial_metrics['financial_data'])
    diag_text = print_diagnostics(diagnostics)
    st.code(diag_text, language="text")

st.divider()

# Quarterly comparison table with LTM
st.subheader("Quarterly performance & LTM analysis")
try:
    quarterly_history = st.session_state.financial_metrics['fetcher'].get_quarterly_history(num_quarters=8)
    
    # Format the dataframe for display
    display_df = quarterly_history.copy()
    
    # Format date - handle both datetime and string
    display_df['Date'] = display_df['Date'].apply(
        lambda x: x if isinstance(x, str) else pd.to_datetime(x).strftime('%Y-%m-%d')
    )
    
    # Format numbers with proper spacing
    for col in ['Total Revenue', 'Net Income', 'Operating Income']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: (
                    f"${x/1e9:.1f}B" if abs(x) >= 1e9 else
                    f"${x/1e6:.0f}M" if abs(x) >= 1e6 else
                    f"${x:,.2f}" if x != 0 else "N/A"
                )
            )
    
    if 'EPS' in display_df.columns:
        display_df['EPS'] = display_df['EPS'].apply(
            lambda x: (
                f"${x:.2f}" if pd.notna(x) else "N/A"
            )
        )
    
    # Display with LTM highlighted
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Show trend analysis
    with st.expander("Quarterly Trend Analysis"):
        if len(quarterly_history) > 2:
            ltm_row = quarterly_history.iloc[0]
            recent_q = quarterly_history.iloc[1]
            prior_q = quarterly_history.iloc[2]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                revenue_val = ltm_row['Total Revenue']
                if revenue_val > 1e9:
                    st.metric("LTM Revenue", f"${revenue_val/1e9:.1f}B")
                elif revenue_val > 1e6:
                    st.metric("LTM Revenue", f"${revenue_val/1e6:.0f}M")
                else:
                    st.metric("LTM Revenue", "N/A")
            
            with col2:
                ni_val = ltm_row['Net Income']
                if ni_val > 1e9:
                    st.metric("LTM Net Income", f"${ni_val/1e9:.1f}B")
                elif ni_val > 1e6:
                    st.metric("LTM Net Income", f"${ni_val/1e6:.0f}M")
                else:
                    st.metric("LTM Net Income", "N/A")
            
            with col3:
                eps_val = ltm_row['EPS']
                if pd.notna(eps_val):
                    st.metric("LTM EPS", f"${eps_val:.2f}")
                else:
                    st.metric("LTM EPS", "N/A")
            
            # Growth metrics
            st.write("**Latest quarter vs previous quarter:**")
            try:
                rev_growth = (
                    (recent_q['Total Revenue'] - prior_q['Total Revenue'])
                    / abs(prior_q['Total Revenue']) * 100
                    if prior_q['Total Revenue'] else None
                )
                ni_growth = (
                    (recent_q['Net Income'] - prior_q['Net Income'])
                    / abs(prior_q['Net Income']) * 100
                    if prior_q['Net Income'] else None
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(
                        f"Revenue Growth: **{rev_growth:+.1f}%**" if rev_growth is not None else
                        "Revenue Growth: **N/A**"
                    )
                with col2:
                    st.write(
                        f"Net Income Growth: **{ni_growth:+.1f}%**" if ni_growth is not None else
                        "Net Income Growth: **N/A**"
                    )
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                st.info("Quarter-over-quarter growth is unavailable for this ticker.")
                
except Exception as e:
    st.warning(f"Could not load quarterly data: {str(e)}")

st.divider()

st.subheader("Annual results: last two fiscal years")
annual_history = st.session_state.financial_metrics.get('annual_history', pd.DataFrame())
if annual_history.empty:
    st.info("Annual statement data is not available for this company.")
else:
    annual_display = annual_history.copy()
    annual_display['Date'] = annual_display['Date'].apply(
        lambda value: value if isinstance(value, str) else pd.to_datetime(value).strftime('%Y-%m-%d')
    )
    for column in ['Total Revenue', 'Net Income', 'Operating Income']:
        if column in annual_display.columns:
            annual_display[column] = annual_display[column].apply(
                lambda value: (
                    f"${value / 1e9:.1f}B" if abs(value) >= 1e9 else
                    f"${value / 1e6:.0f}M" if abs(value) >= 1e6 else
                    f"${value:,.0f}"
                )
            )
    if 'EPS' in annual_display.columns:
        annual_display['EPS'] = annual_display['EPS'].apply(
            lambda value: f"${value:.2f}" if pd.notna(value) else "N/A"
        )
    st.dataframe(annual_display, use_container_width=True, hide_index=True)
    st.caption("Source: Yahoo Finance annual financial statements. Verify material claims against official filings.")

st.divider()

# 5-Year Candlestick Chart with Bollinger Bands and SMA
st.subheader("5-Year Price Chart with Bollinger Bands and SMA")

try:
    # Fetch 5-year historical weekly data
    ticker_obj = st.session_state.financial_metrics['fetcher'].stock
    hist = ticker_obj.history(period="5y", interval="1wk")
    
    # Fallback: if weekly data is empty, try daily and resample to weekly
    if hist is None or len(hist) == 0:
        st.info("Trying daily data and resampling to weekly...")
        hist = ticker_obj.history(period="5y")
        if hist is not None and len(hist) > 0:
            hist = hist.resample('W').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            })
            # Remove NaN rows from aggregation
            hist = hist.dropna(subset=['Open', 'Close'])
    
    # Second fallback: try 3-year data if 5-year is insufficient
    if hist is None or len(hist) < 20:
        st.info("Not enough 5-year data, trying 3-year data...")
        hist = ticker_obj.history(period="3y", interval="1wk")
        if hist is None or len(hist) == 0:
            hist = ticker_obj.history(period="3y")
            if hist is not None and len(hist) > 0:
                hist = hist.resample('W').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                })
                hist = hist.dropna(subset=['Open', 'Close'])
    
    if hist is None or len(hist) < 20:
        st.warning(f"Not enough historical data available for this ticker (got {len(hist) if hist is not None else 0} weeks, need at least 20)")
    else:
        # Add technical indicators using pandas_ta_classic
        df = hist.copy()
        
        # Calculate Bollinger Bands properly (for weekly data: 20 weeks)
        bb = ta.bbands(df['Close'], length=20, std=2)
        if bb is not None and len(bb) > 0:
            df['BB_Upper'] = bb.iloc[:, 2] if bb.shape[1] > 2 else bb.iloc[:, 0]
            df['BB_Middle'] = bb.iloc[:, 1] if bb.shape[1] > 1 else df['Close'].rolling(20).mean()
            df['BB_Lower'] = bb.iloc[:, 0]
        else:
            # Fallback manual calculation
            df['BB_Middle'] = df['Close'].rolling(20).mean()
            df['BB_Std'] = df['Close'].rolling(20).std()
            df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
            df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
        
        # Calculate SMA 50 (for weekly data: 50 weeks)
        if len(df) >= 50:
            sma = ta.sma(df['Close'], length=50)
            df['SMA_50'] = sma if sma is not None else df['Close'].rolling(50).mean()
        else:
            df['SMA_50'] = df['Close'].rolling(min(20, len(df))).mean()
        
        # Create candlestick chart
        fig = go.Figure()
        
        # Add candlestick trace
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC'
        ))
        
        # Add Bollinger Bands with better colors for light/dark mode
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_Upper'],
            mode='lines',
            name='BB Upper (20,2)',
            line=dict(color='#FF6B6B', width=1.5, dash='dot')
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_Lower'],
            mode='lines',
            name='BB Lower (20,2)',
            line=dict(color='#FF6B6B', width=1.5, dash='dot'),
            fill='tonexty',
            fillcolor='rgba(255, 107, 107, 0.1)'
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_Middle'],
            mode='lines',
            name='BB Middle (SMA 20)',
            line=dict(color='#FFA500', width=2)
        ))
        
        # Add SMA 50 - green for clarity
        fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA_50'],
            mode='lines',
            name='SMA 50',
            line=dict(color='#2ECC71', width=2.5)
        ))
        
        fig.update_layout(
            title=f"5-Year Weekly Price Chart - {st.session_state.ticker}",
            yaxis_title="Price (USD)",
            xaxis_title="Date",
            template="plotly_dark",
            hovermode='x unified',
            height=600,
            xaxis_rangeslider_visible=False,
            font=dict(size=11)
        )
        
        st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.warning(f"Could not generate chart: {str(e)}")

# Generate initial summary automatically
if st.session_state.ticker_data_loaded and not st.session_state.initial_summary_generated:
    with st.spinner("Generating AI summary..."):
        try:
            quarterly_df = st.session_state.financial_metrics['quarterly_history'].copy()
            annual_df = st.session_state.financial_metrics.get('annual_history', pd.DataFrame())
            profile = st.session_state.financial_metrics.get('company_profile', {})
            evidence_context = [
                f"Company profile: {profile.get('name', st.session_state.ticker)}; "
                f"sector={profile.get('sector', 'N/A')}; industry={profile.get('industry', 'N/A')}; "
                f"business summary={profile.get('business_summary', 'N/A')}",
                "Observed key metrics:\n" + "\n".join(
                    f"- {name}: {value}"
                    for name, value in st.session_state.financial_metrics['formatted'].items()
                ),
                "Quarterly history (up to 8 quarters):\n" + quarterly_df.to_string(index=False),
                "Annual history (up to 2 fiscal years):\n" + (
                    annual_df.to_string(index=False) if not annual_df.empty else "N/A"
                ),
            ]
            evidence_context = "\n\n".join(evidence_context)

            summary_sources = []
            if st.session_state.rag_system:
                summary_result = st.session_state.rag_system.query(
                    "Assess the company's financial health, two-year trend, valuation, risks, and data limitations.",
                    st.session_state.llm_client,
                )
                summary_response = summary_result['answer']
                summary_sources = summary_result.get('source_types', [])
            else:
                summary_response = st.session_state.llm_client.generate_financial_analysis(
                    st.session_state.ticker,
                    st.session_state.financial_metrics['formatted'],
                    evidence_context,
                )

            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": summary_response,
                "sources": summary_sources,
            })
            st.session_state.initial_summary_generated = True
            st.rerun()
        except Exception as e:
            st.warning(f"Could not generate summary: {str(e)}")

st.divider()

# CHAT SECTION
st.subheader("Ask Questions About This Analysis")

# Display conversation history
for message in st.session_state.conversation_history:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message['content'])
    else:
        with st.chat_message("assistant"):
            st.write(message['content'])
            sources = message.get('sources', [])
            if sources:
                st.caption("Retrieved evidence: " + ", ".join(sorted(set(sources))))

# Input area for new question
user_question = st.chat_input("Ask a follow-up question...")

if user_question:
    # Add user message to history
    st.session_state.conversation_history.append({
        "role": "user",
        "content": user_question
    })
    
    # Get LLM response using RAG
    with st.spinner("Generating response with financial data..."):
        try:
            source_types = []
            if st.session_state.rag_system:
                # Use RAG for retrieval-augmented response
                rag_result = st.session_state.rag_system.query(
                    user_question,
                    st.session_state.llm_client
                )
                response = rag_result["answer"]
                source_types = rag_result.get("source_types", [])
            else:
                # Fallback to direct LLM if RAG not available
                response = st.session_state.llm_client.answer_question(
                    user_question,
                    st.session_state.ticker,
                    st.session_state.financial_metrics['formatted']
                )
        except Exception as e:
            response = f"Error: {str(e)}\n\nPlease check your API key and internet connection."
    
    # Add assistant response to history
    st.session_state.conversation_history.append({
        "role": "assistant",
        "content": response,
        "sources": source_types,
    })
    
    # Rerun to show both messages cleanly
    st.rerun()
