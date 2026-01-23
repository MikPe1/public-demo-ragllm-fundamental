"""
LLM Integration for multiple providers
"""

from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
import os


class LLMIntegrator:
    """
    Integrates with multiple LLM providers via LangChain
    """
    
    def __init__(self, provider: str, api_key: str):
        self.provider = provider
        self.api_key = api_key
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize LLM client based on provider"""
        
        if self.provider == "OpenAI":
            return ChatOpenAI(
                model="gpt-4",
                api_key=self.api_key,
                temperature=0.7
            )
        
        elif self.provider == "DeepSeek":
            # DeepSeek is compatible with OpenAI API
            return ChatOpenAI(
                model="deepseek-chat",
                api_key=self.api_key,
                base_url="https://api.deepseek.com/v1",
                temperature=0.7
            )
        
        elif self.provider == "Ollama (local)":
            return ChatOllama(
                model="llama2",
                temperature=0.7
            )
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def generate_financial_analysis(self, ticker: str, metrics: dict, context: str) -> str:
        """
        Generate analysis based on financial metrics
        
        Args:
            ticker: Stock ticker
            metrics: Financial metrics dictionary
            context: Additional context about the stock
        
        Returns:
            Analysis text from LLM
        """
        
        system_prompt = f"""You are a professional financial analyst specializing in equity research.
Analyze the following financial metrics for {ticker} and provide insights.

Financial Metrics:
{self._format_metrics(metrics)}

Provide a concise analysis focusing on:
1. Financial health assessment
2. Key strengths and concerns
3. Valuation assessment
4. Brief investment perspective

Keep your response professional but accessible."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=context or f"Provide a financial analysis for {ticker}")
        ]
        
        try:
            response = self.client.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error generating analysis: {str(e)}"
    
    def answer_question(self, question: str, ticker: str, metrics: dict) -> str:
        """
        Answer a question about financial analysis
        
        Args:
            question: User's question
            ticker: Stock ticker
            metrics: Financial metrics dictionary
        
        Returns:
            Answer from LLM
        """
        
        system_prompt = f"""You are a professional financial analyst.
Answer questions about {ticker}'s financial analysis based on these metrics:

{self._format_metrics(metrics)}

Be concise, accurate, and professional in your responses."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ]
        
        try:
            response = self.client.invoke(messages)
            return response.content
        except Exception as e:
            error_msg = str(e)
            if "Connection" in error_msg or "401" in error_msg or "403" in error_msg:
                return f"Authentication or connection error: Please verify your API key for {self.provider}.\n\nDetails: {error_msg}"
            elif "timeout" in error_msg.lower():
                return f"Request timeout: The API server is taking too long to respond. Please try again."
            else:
                return f"Error answering question: {error_msg}"
    
    @staticmethod
    def _format_metrics(metrics: dict) -> str:
        """Format metrics dictionary for LLM prompt"""
        formatted = []
        for key, value in metrics.items():
            formatted.append(f"- {key}: {value}")
        return "\n".join(formatted)
    
    def chat(self, conversation_history: list) -> str:
        """
        Continue a multi-turn conversation
        
        Args:
            conversation_history: List of {"role": "user"/"assistant", "content": "..."} dicts
        
        Returns:
            LLM response
        """
        
        messages = []
        
        # Convert conversation history to LangChain messages
        for msg in conversation_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                # In LangChain, we use SystemMessage or just track context
                pass
        
        try:
            response = self.client.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error in conversation: {str(e)}"
