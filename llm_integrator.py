"""
LLM Integration for multiple providers
"""

from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class LLMIntegrator:
    """
    Integrates with multiple LLM providers via LangChain
    """
    
    def __init__(self, provider: str, api_key: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize LLM client based on provider"""

        if self.provider == "OpenAI":
            if not self.api_key:
                raise ValueError("An OpenAI API key is required.")
            return ChatOpenAI(
                model="gpt-4o-mini",
                api_key=self.api_key,
                temperature=0.7
            )

        elif self.provider == "DeepSeek":
            if not self.api_key:
                raise ValueError("A DeepSeek API key is required.")
            # DeepSeek is compatible with OpenAI API
            return ChatOpenAI(
                model="deepseek-chat",
                api_key=self.api_key,
                base_url="https://api.deepseek.com/v1",
                temperature=0.7
            )

        elif self.provider == "Ollama (local)":
            try:
                from langchain_ollama import ChatOllama
            except ImportError as error:
                raise RuntimeError(
                    "Ollama integration is unavailable. Install the "
                    "langchain-ollama package or select OpenAI/DeepSeek."
                ) from error

            return ChatOllama(model="llama3.2", temperature=0.7)
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def invoke(self, messages):
        """Expose the LangChain invocation contract to integrations such as RAG."""
        return self.client.invoke(messages)
    
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
        
        system_prompt = f"""You are a professional equity-research analyst.
    Assess {ticker} using only the financial metrics and evidence supplied below.
    Do not invent facts, forecasts, or company history. Clearly label missing data.

Financial Metrics:
{self._format_metrics(metrics)}

    Evidence and retrieved documents:
    {context or 'No additional evidence was supplied.'}

    Return a research-oriented assessment with these headings:
    1. Evidence-based conclusion: attractive, mixed, weak, or insufficient data
    2. Financial health and two-year trend
    3. Valuation and key risks
    4. Data limitations and what should be verified in official filings

    Do not present this as personalized investment advice or certainty about future returns."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Assess the investment case for {ticker} based on the evidence above.")
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
    Answer questions about {ticker} using only these observed metrics:

{self._format_metrics(metrics)}

    If the data is insufficient, say so. Do not invent a value or imply personalized investment advice."""
        
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
                messages.append(AIMessage(content=msg["content"]))
        
        try:
            response = self.client.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error in conversation: {str(e)}"
