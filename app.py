import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pandas_ta_classic as ta
from financial_calculator import FinancialCalculator
from data_fetcher import YahooFinanceFetcher
from llm_integrator import LLMIntegrator
from diagnostics import diagnose_financial_data, print_diagnostics
from rag_system import FinancialRAGSystem

st.set_page_config(page_title="LLM-Powered Fundamental Analysis (RAG-Based)", layout="wide")
st.title("LLM-Powered Fundamental Analysis (RAG-Based)")

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
if "initial_summary_generated" not in st.session_state:
    st.session_state.initial_summary_generated = False
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None

# Step 1: API Configuration
st.header("Step 1: API Configuration")
col1, col2 = st.columns(2)

with col1:
    provider = st.selectbox(
        "Select LLM Provider:",
        ("OpenAI", "DeepSeek", "Ollama (local)"),
        key="provider_select"
    )

with col2:
    api_key = st.text_input(
        f"Paste your API key for {provider}",
        type="password",
        key="api_key_input",
        help="Don't worry I don't store them i promise! :)"
    )

if api_key and provider:
    st.session_state.api_key = api_key
    st.session_state.provider = provider
    
    # Initialize LLM client
    try:
        if st.session_state.llm_client is None:
            st.session_state.llm_client = LLMIntegrator(provider, api_key)
        st.success("API credentials configured successfully")
    except Exception as e:
        st.error(f"Failed to initialize LLM: {str(e)}")
else:
    st.warning("Please provide both LLM provider and API key to continue")
    st.stop()

# Step 2: Stock Ticker Input
st.header("Step 2: Select Stock")
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
                quarterly_history = fetcher.get_quarterly_history(num_quarters=4)
                
                st.session_state.financial_metrics = {
                    'raw': metrics,
                    'formatted': formatted_metrics,
                    'financial_data': financial_data,
                    'fetcher': fetcher,
                    'quarterly_history': quarterly_history
                }
                
                # Build RAG system
                if st.session_state.api_key:
                    try:
                        rag_system = FinancialRAGSystem(
                            st.session_state.api_key,
                            provider=st.session_state.provider
                        )
                        rag_system.build_knowledge_base(
                            ticker,
                            financial_data,
                            quarterly_history
                        )
                        st.session_state.rag_system = rag_system
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
st.subheader("Quarterly Performance & LTM Analysis")
try:
    quarterly_history = st.session_state.financial_metrics['fetcher'].get_quarterly_history(num_quarters=4)
    
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
                f"${x:.2f}" if (isinstance(x, (int, float)) and x > 0 and x == x) else "N/A"
            )
        )
    
    # Display with LTM highlighted
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Show trend analysis
    with st.expander("Quarterly Trend Analysis"):
        if len(quarterly_history) > 1:
            ltm_row = quarterly_history.iloc[0]
            recent_q = quarterly_history.iloc[1]
            
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
                if isinstance(eps_val, (int, float)) and eps_val > 0 and eps_val == eps_val:  # NaN check
                    st.metric("LTM EPS", f"${eps_val:.2f}")
                else:
                    st.metric("LTM EPS", "N/A")
            
            # Growth metrics
            st.write("**Recent Quarter vs LTM:**")
            try:
                rev_growth = ((recent_q['Total Revenue'] - ltm_row['Total Revenue']) / ltm_row['Total Revenue'] * 100) if ltm_row['Total Revenue'] > 0 else 0
                ni_growth = ((recent_q['Net Income'] - ltm_row['Net Income']) / ltm_row['Net Income'] * 100) if ltm_row['Net Income'] > 0 else 0
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"Revenue Growth: **{rev_growth:+.1f}%**")
                with col2:
                    st.write(f"Net Income Growth: **{ni_growth:+.1f}%**")
            except:
                pass
                
except Exception as e:
    st.warning(f"Could not load quarterly data: {str(e)}")

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
            # Prepare quarterly context
            quarterly_df = st.session_state.financial_metrics['quarterly_history'].copy()
            quarterly_text = "Quarterly Performance (LTM + Last 4 Quarters):\n"
            for idx, row in quarterly_df.iterrows():
                quarterly_text += f"- {row['Date']}: Revenue ${row['Total Revenue']/1e9:.1f}B, Net Income ${row['Net Income']/1e9:.1f}B, EPS ${row['EPS']:.2f}\n"
            
            summary_prompt = f"""Provide a concise financial summary for {st.session_state.ticker}.

Key Metrics (LTM - Last Twelve Months):
{quarterly_text}

Focus on:
1. Current valuation (P/E: {st.session_state.financial_metrics['formatted'].get('P/E Ratio', 'N/A')})
2. Profitability (Operating Margin: {st.session_state.financial_metrics['formatted'].get('Operating Margin', 'N/A')}, ROE: {st.session_state.financial_metrics['formatted'].get('ROE', 'N/A')})
3. Financial strength (Debt-to-Capital: {st.session_state.financial_metrics['formatted'].get('Debt-to-Capital', 'N/A')}, Interest Coverage: {st.session_state.financial_metrics['formatted'].get('Interest Coverage', 'N/A')})
4. Quarterly trends (growth/decline)
5. Brief investment perspective"""
            
            summary_response = st.session_state.llm_client.answer_question(
                summary_prompt,
                st.session_state.ticker,
                st.session_state.financial_metrics['formatted']
            )
            
            # Add to conversation history
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": summary_response
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
            if st.session_state.rag_system:
                # Use RAG for retrieval-augmented response
                rag_result = st.session_state.rag_system.query(
                    user_question,
                    st.session_state.llm_client
                )
                response = rag_result["answer"]
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
        "content": response
    })
    
    # Rerun to show both messages cleanly
    st.rerun()
