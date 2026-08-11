"""
RAG (Retrieval-Augmented Generation) System for Financial Analysis
Uses FAISS for vector storage and LangChain for retrieval
"""

from typing import List, Dict, Optional, Any
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage


class FinancialRAGSystem:
    """
    RAG system for financial data retrieval and analysis
    Supports both OpenAI and HuggingFace embeddings for Streamlit Cloud compatibility
    """
    
    def __init__(self, api_key: str, provider: str = "OpenAI"):
        self.api_key = api_key
        self.provider = provider
        self.vector_store = None
        self.retriever = None
        
        try:
            self.embeddings = self._initialize_embeddings()
        except Exception as e:
            import streamlit as st
            st.error(f"⚠️ RAG Initialization Error: {e}")
            st.info("Using non-RAG chat mode. Try refreshing or switching providers.")
            self.embeddings = None
        self.qa_chain = None
    
    def _initialize_embeddings(self):
        """
        Initialize embeddings based on provider and environment
        Supports both Streamlit Cloud and local deployment
        - OpenAI: Uses OpenAI Embeddings (requires API key)
        - DeepSeek/Ollama/Cloud: Uses HuggingFace (local, free, cached)
        """
        try:
            if self.provider == "OpenAI":
                # Try OpenAI embeddings first
                return OpenAIEmbeddings(api_key=self.api_key, request_timeout=10)
            else:
                # Use HuggingFace for DeepSeek, Ollama, and Streamlit Cloud
                try:
                    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
                except ImportError:
                    # Fallback if langchain-huggingface not installed
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                
                return HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    show_progress=False
                )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embeddings: {str(e)}")
    
    def build_knowledge_base(
        self,
        ticker: str,
        financial_data: Dict,
        quarterly_history: pd.DataFrame,
        annual_history: Optional[pd.DataFrame] = None,
        company_profile: Optional[Dict] = None,
    ) -> None:
        """
        Build RAG knowledge base from financial data
        
        Args:
            ticker: Stock ticker
            financial_data: Dictionary with financial metrics
            quarterly_history: DataFrame with quarterly data
        """
        
        if self.embeddings is None:
            print("Embeddings not available - RAG will be skipped")
            return
        
        try:
            # Create documents from financial data
            documents = self._create_documents(
                ticker,
                financial_data,
                quarterly_history,
                annual_history,
                company_profile,
            )
            
            # Split documents into chunks for better retrieval
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                separators=["\n\n", "\n", " ", ""]
            )
            
            split_docs = text_splitter.split_documents(documents)
            
            # Create FAISS vector store
            self.vector_store = FAISS.from_documents(split_docs, self.embeddings)
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        except Exception as e:
            print(f"Warning: Could not build knowledge base: {e}")
            self.vector_store = None
            self.retriever = None

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float, suffix: str = '') -> str:
        """Format a ratio without turning missing data into a fake zero."""
        if numerator is None or denominator in (None, 0):
            return 'N/A'
        try:
            numerator = float(numerator)
            denominator = float(denominator)
            if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
                return 'N/A'
            return f'{numerator / denominator:.2f}{suffix}'
        except (TypeError, ValueError):
            return 'N/A'

    def _create_documents(
        self,
        ticker: str,
        financial_data: Dict,
        quarterly_history: pd.DataFrame,
        annual_history: Optional[pd.DataFrame] = None,
        company_profile: Optional[Dict] = None,
    ) -> List[Document]:
        """
        Create LangChain documents from financial data
        """
        
        documents = []
        annual_history = annual_history if annual_history is not None else pd.DataFrame()
        company_profile = company_profile or {}

        profile_text = f"""
Company profile for {ticker}:
- Name: {company_profile.get('name', 'N/A')}
- Sector: {company_profile.get('sector', 'N/A')}
- Industry: {company_profile.get('industry', 'N/A')}
- Country: {company_profile.get('country', 'N/A')}
- Employees: {company_profile.get('employees', 'N/A')}
- Business summary: {company_profile.get('business_summary', 'N/A')}

Source: Yahoo Finance company profile. Treat qualitative information as context,
not as independently verified evidence of investment quality.
"""
        documents.append(Document(
            page_content=profile_text,
            metadata={'type': 'company_profile', 'ticker': ticker, 'source': 'Yahoo Finance'},
        ))

        if not annual_history.empty:
            annual_text = f"""
ANNUAL FINANCIAL STATEMENTS SUMMARY FOR {ticker}:
The following values are reported annual figures, ordered from oldest to newest.
"""
            for _, row in annual_history.iterrows():
                annual_text += (
                    f"\n{row['Date']}: Revenue ${row['Total Revenue']:,.0f}; "
                    f"Net Income ${row['Net Income']:,.0f}; "
                    f"Operating Income ${row['Operating Income']:,.0f}; "
                    f"EPS ${row['EPS']:.2f}"
                )
            documents.append(Document(
                page_content=annual_text,
                metadata={'type': 'annual_statements', 'ticker': ticker, 'source': 'Yahoo Finance'},
            ))
        
        # 1. VALUATION DOCUMENT
        shares_outstanding = financial_data.get('shares_outstanding', 0)
        eps = (
            financial_data.get('net_income', 0) / shares_outstanding
            if shares_outstanding else None
        )
        valuation_pe = self._safe_ratio(financial_data.get('stock_price', 0), eps)
        valuation_text = f"""
Stock: {ticker}

VALUATION METRICS:
- P/E Ratio (LTM): {valuation_pe}
- Price-to-Book: Market Cap {financial_data.get('market_cap', 0):.2e} vs Shareholders Equity {financial_data.get('shareholders_equity', 0):.2e}
- EV/EBIT: Enterprise Value {financial_data.get('market_cap', 0) + financial_data.get('total_debt', 0) - financial_data.get('cash', 0):.2e} / EBIT {financial_data.get('ebit', 0):.2e}
- EV/EBITDA Consideration: Based on operating efficiency

Investment perspective: Higher ratios may indicate growth expectations or market premium.
"""
        documents.append(Document(page_content=valuation_text, metadata={"type": "valuation", "ticker": ticker}))

        revenue = financial_data.get('total_revenue', 0)
        equity = financial_data.get('shareholders_equity', 0)
        total_capital = financial_data.get('total_capital', 0)
        ebit = financial_data.get('ebit', 0)
        interest_expense = financial_data.get('interest_expense', 0)
        current_liabilities = financial_data.get('current_liabilities', 0)
        operating_margin = self._safe_ratio(
            financial_data.get('operating_income', 0), revenue, '%'
        )
        roe = self._safe_ratio(financial_data.get('net_income', 0), equity, '%')
        gross_margin = self._safe_ratio(financial_data.get('gross_profit', 0), revenue, '%')
        debt_to_capital = self._safe_ratio(
            financial_data.get('total_debt', 0), total_capital, '%'
        )
        interest_coverage = self._safe_ratio(ebit, abs(interest_expense), 'x')
        quick_ratio = self._safe_ratio(
            financial_data.get('cash', 0) + financial_data.get('accounts_receivable', 0),
            current_liabilities,
            'x',
        )

        # 2. PROFITABILITY DOCUMENT
        profitability_text = f"""
Stock: {ticker}

PROFITABILITY ANALYSIS (LTM - Last Twelve Months):
- Net Income: ${financial_data.get('net_income', 0):,.0f}
- Operating Income: ${financial_data.get('operating_income', 0):,.0f}
- Operating Margin: {operating_margin}
- Return on Equity (ROE): {roe}
- Gross Margin: {gross_margin}

Profitability indicates company's efficiency and pricing power.
Consistent high margins suggest competitive advantages.
"""
        documents.append(Document(page_content=profitability_text, metadata={"type": "profitability", "ticker": ticker}))
        
        # 3. FINANCIAL HEALTH DOCUMENT
        health_text = f"""
Stock: {ticker}

FINANCIAL HEALTH & SOLVENCY:
- Total Debt: ${financial_data.get('total_debt', 0):,.0f}
- Cash Position: ${financial_data.get('cash', 0):,.0f}
- Net Debt: ${financial_data.get('total_debt', 0) - financial_data.get('cash', 0):,.0f}
- Debt-to-Capital Ratio: {debt_to_capital}
- Interest Coverage Ratio: {interest_coverage}
- Quick Ratio: {quick_ratio}

Strong metrics: Low leverage, high coverage, solid liquidity.
Weak metrics: High debt, low coverage, potential refinancing risk.
"""
        documents.append(Document(page_content=health_text, metadata={"type": "financial_health", "ticker": ticker}))
        
        # 4. QUARTERLY TRENDS DOCUMENT
        if quarterly_history is not None and not quarterly_history.empty:
            trends_text = f"""
Stock: {ticker}

QUARTERLY PERFORMANCE TRENDS:
"""
            for idx, row in quarterly_history.iterrows():
                period = row['Date']
                revenue = row['Total Revenue']
                net_income = row['Net Income']
                eps = row['EPS']
                
                trends_text += f"\n{period}:"
                trends_text += f"\n  Revenue: ${revenue:,.0f}"
                trends_text += f"\n  Net Income: ${net_income:,.0f}"
                trends_text += f"\n  EPS: ${eps:.2f}" if pd.notna(eps) else f"\n  EPS: N/A"
            
            # Calculate trend direction
            if len(quarterly_history) >= 2:
                ltm_revenue = quarterly_history.iloc[0]['Total Revenue']
                recent_revenue = quarterly_history.iloc[1]['Total Revenue']
                trend = ((recent_revenue - ltm_revenue) / ltm_revenue * 100) if ltm_revenue > 0 else 0
                trends_text += f"\n\nTrend Direction: {trend:+.1f}% (Recent vs LTM)"
            
            documents.append(Document(page_content=trends_text, metadata={"type": "quarterly_trends", "ticker": ticker}))
        
        # 5. REVENUE ANALYSIS DOCUMENT
        revenue_text = f"""
Stock: {ticker}

REVENUE ANALYSIS:
- Total Revenue (LTM): ${financial_data.get('total_revenue', 0):,.0f}
- Cost of Revenue: ${financial_data.get('cost_of_revenue', 0):,.0f}
- Gross Profit: ${financial_data.get('gross_profit', 0):,.0f}
- Gross Margin: {(financial_data.get('gross_profit', 0) / financial_data.get('total_revenue', 1) * 100):.1f}%

Revenue quality assessment:
- Consistent revenue growth indicates strong market position
- High gross margins suggest pricing power and cost efficiency
- Declining revenue may indicate market challenges or competition
"""
        documents.append(Document(page_content=revenue_text, metadata={"type": "revenue_analysis", "ticker": ticker}))
        
        return documents
    
    def get_rag_chain(self, llm_client) -> Any:
        """
        Get RAG retriever for question answering
        
        Args:
            llm_client: LangChain LLM client
        
        Returns:
            Retriever object
        """
        
        if self.retriever is None:
            raise ValueError("Knowledge base not built. Call build_knowledge_base() first.")
        
        return self.retriever
    
    def query(self, question: str, llm_client) -> Dict:
        """
        Query the RAG system
        
        Args:
            question: User question
            llm_client: LangChain LLM client
        
        Returns:
            Dictionary with answer and source documents
        """
        
        try:
            # Retrieve relevant documents
            retrieved_docs = self.retriever.invoke(question)
            
            # Format documents as context
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # Create messages for LLM
            system_message = SystemMessage(content="""You are an equity-research analyst assisting with a financial data portfolio project.
Use only the supplied retrieved documents as factual evidence. If the documents do not
contain enough information, say so explicitly instead of guessing. Separate historical
facts from interpretation, cite the document type in your reasoning, and call out data
quality limitations. You may provide a research-oriented assessment, but do not present
it as personalized investment advice or certainty about future returns.
Be concise but comprehensive.""")
            
            user_message = HumanMessage(content=f"""Financial Documents Context:
{context}

User Question: {question}

Please provide a detailed analysis based on the financial data provided.""")
            
            # Get response from LLM
            response = llm_client.invoke([system_message, user_message])
            
            return {
                "answer": response.content,
                "sources": retrieved_docs,
                "source_types": [doc.metadata.get('type', 'unknown') for doc in retrieved_docs],
            }
        except Exception as e:
            return {
                "answer": f"Error retrieving information: {str(e)}",
                "sources": []
            }
