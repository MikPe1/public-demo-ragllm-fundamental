"""
RAG (Retrieval-Augmented Generation) System for Financial Analysis
Uses FAISS for vector storage and LangChain for retrieval
"""

from typing import List, Dict, Optional, Any
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
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
    
    def build_knowledge_base(self, ticker: str, financial_data: Dict, quarterly_history: pd.DataFrame) -> None:
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
            documents = self._create_documents(ticker, financial_data, quarterly_history)
            
            # Split documents into chunks for better retrieval
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                separators=["\n\n", "\n", " ", ""]
            )
            
            split_docs = text_splitter.split_documents(documents)
            
            # Create FAISS vector store
            self.vector_store = FAISS.from_documents(split_docs, self.embeddings)
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        except Exception as e:
            print(f"Warning: Could not build knowledge base: {e}")
            self.vector_store = None
            self.retriever = None
    
    def _create_documents(self, ticker: str, financial_data: Dict, quarterly_history: pd.DataFrame) -> List[Document]:
        """
        Create LangChain documents from financial data
        """
        
        documents = []
        
        # 1. VALUATION DOCUMENT
        valuation_text = f"""
Stock: {ticker}

VALUATION METRICS:
- P/E Ratio (LTM): {financial_data.get('stock_price', 0):.2f} / {financial_data.get('net_income', 0) / financial_data.get('shares_outstanding', 1):.2f}
- Price-to-Book: Market Cap {financial_data.get('market_cap', 0):.2e} vs Shareholders Equity {financial_data.get('shareholders_equity', 0):.2e}
- EV/EBIT: Enterprise Value {financial_data.get('market_cap', 0) + financial_data.get('total_debt', 0) - financial_data.get('cash', 0):.2e} / EBIT {financial_data.get('ebit', 0):.2e}
- EV/EBITDA Consideration: Based on operating efficiency

Investment perspective: Higher ratios may indicate growth expectations or market premium.
"""
        documents.append(Document(page_content=valuation_text, metadata={"type": "valuation", "ticker": ticker}))
        
        # 2. PROFITABILITY DOCUMENT
        profitability_text = f"""
Stock: {ticker}

PROFITABILITY ANALYSIS (LTM - Last Twelve Months):
- Net Income: ${financial_data.get('net_income', 0):,.0f}
- Operating Income: ${financial_data.get('operating_income', 0):,.0f}
- Operating Margin: {(financial_data.get('operating_income', 0) / financial_data.get('total_revenue', 1) * 100):.1f}%
- Return on Equity (ROE): {(financial_data.get('net_income', 0) / financial_data.get('shareholders_equity', 1) * 100):.1f}%
- Gross Margin: {(financial_data.get('gross_profit', 0) / financial_data.get('total_revenue', 1) * 100):.1f}%

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
- Debt-to-Capital Ratio: {(financial_data.get('total_debt', 0) / financial_data.get('total_capital', 1) * 100):.1f}%
- Interest Coverage Ratio: {(financial_data.get('ebit', 0) / financial_data.get('interest_expense', 1)):.2f}x
- Quick Ratio: {((financial_data.get('cash', 0) + financial_data.get('accounts_receivable', 0)) / financial_data.get('current_liabilities', 1)):.2f}x

Strong metrics: Low leverage, high coverage, solid liquidity.
Weak metrics: High debt, low coverage, potential refinancing risk.
"""
        documents.append(Document(page_content=health_text, metadata={"type": "financial_health", "ticker": ticker}))
        
        # 4. QUARTERLY TRENDS DOCUMENT
        if not quarterly_history.empty:
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
                trends_text += f"\n  EPS: ${eps:.2f}" if isinstance(eps, (int, float)) else f"\n  EPS: N/A"
            
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
            system_message = SystemMessage(content="""You are a professional financial analyst specializing in equity research.
Use the provided financial documents to answer questions with specific, data-driven insights.
Reference specific metrics and trends from the documents.
Be concise but comprehensive.""")
            
            user_message = HumanMessage(content=f"""Financial Documents Context:
{context}

User Question: {question}

Please provide a detailed analysis based on the financial data provided.""")
            
            # Get response from LLM
            response = llm_client.invoke([system_message, user_message])
            
            return {
                "answer": response.content,
                "sources": retrieved_docs
            }
        except Exception as e:
            return {
                "answer": f"Error retrieving information: {str(e)}",
                "sources": []
            }
