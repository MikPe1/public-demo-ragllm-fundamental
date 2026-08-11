# Fundamental Research Copilot

A Streamlit application for evidence-based fundamental equity research. The app combines structured financial data, deterministic financial ratios, company context, and retrieval-augmented generation (RAG) to help analyze a public company.

> This is a research and portfolio project, not investment advice. Yahoo Finance data should be checked against the company's official filings before relying on any material conclusion.

## What It Does

For a ticker, the application:

- retrieves company profile information and financial statements through `yfinance`;
- calculates LTM metrics such as EPS, P/E, ROE, debt-to-capital, interest coverage, EV/EBIT, operating margin, and quick ratio;
- displays up to two fiscal years of annual results and up to eight quarters of history;
- builds an in-memory FAISS index from company context, annual statements, quarterly results, valuation, profitability, financial health, and revenue documents;
- retrieves relevant evidence for an LLM response and displays the document types used;
- supports OpenAI, DeepSeek, and a local Ollama model;
- provides a five-year weekly price chart with Bollinger Bands and a moving average.

## Architecture

```text
Yahoo Finance
     |
     v
Data fetcher -> financial calculator -> Streamlit views
     |
     v
Document builder -> embeddings -> FAISS retriever -> LLM assessment
```

The current RAG corpus is built at runtime for the selected ticker and is not persisted between sessions. OpenAI uses OpenAI embeddings; DeepSeek and Ollama use the local `sentence-transformers/all-MiniLM-L6-v2` embedding model.

## Run Locally

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Choose an LLM provider in the application:

- **OpenAI**: provide an OpenAI API key.
- **DeepSeek**: provide a DeepSeek API key.
- **Ollama (local)**: start Ollama locally and make sure the `llama3.2` model is available. No API key is required.

```powershell
ollama pull llama3.2
```

The first non-OpenAI RAG run downloads the Hugging Face embedding model. This can take longer than subsequent runs.

## Streamlit Cloud

1. Open Streamlit Community Cloud and create a new app.
2. Select `MikPe1/public-demo-ragllm-fundamental`.
3. Use branch `main` and file `app.py`.
4. Install dependencies from `requirements.txt`.
5. Provide an OpenAI or DeepSeek key through the app configuration workflow.

Ollama is intended for local development and is not available as a service on Streamlit Cloud. After changing dependencies, trigger a fresh deployment so the environment installs the current LangChain packages.

## Is RAG Appropriate Here?

Yes, but in its current form it is best understood as a transparent portfolio demonstration rather than a production-grade financial research system.

RAG is useful because the model receives only the retrieved documents relevant to a question, and the UI can expose which document types were retrieved. It also gives one consistent interface for qualitative company context and structured financial history. Without retrieval, the current LLM fallback sees only a compact metrics dictionary and has less context for explaining trends and limitations.

The current corpus is small and generated from structured Yahoo Finance data. For a handful of short documents, deterministic calculations and a well-designed prompt would often be simpler and cheaper than vector search. RAG becomes materially more valuable when the corpus includes primary documents such as 10-K and 10-Q filings, earnings releases, annual reports, risk-factor sections, management discussion, and dated company news.

## Important Limitations

- Yahoo Finance is an aggregation source, not the company's official filing repository.
- The application does not yet download or parse SEC/EDGAR filings, earnings-call transcripts, or dated news.
- The FAISS index is in-memory and recreated for each ticker/session.
- There is no peer-group benchmark, valuation model, forecast, backtest, or deterministic investment score.
- The LLM can explain evidence and identify risks, but it should not be treated as the decision-maker or as a source of facts outside the retrieved context.
- Missing statement line items are reported as unavailable; they should not be interpreted as zero.

## Recommended Next Steps

To make this a stronger senior data scientist portfolio project:

1. Add an official filing ingestion layer using SEC/EDGAR documents with filing dates and source URLs.
2. Keep point-in-time versions of documents to avoid look-ahead bias.
3. Store citations at chunk level and show the exact filing, section, and date behind each claim.
4. Add deterministic trend and quality checks before the LLM step, then evaluate retrieval and answer faithfulness with a small labeled question set.
5. Add peer comparison and a clearly separated research conclusion instead of asking the LLM to make an unconstrained buy/sell decision.
6. Add unit tests for statement parsing, negative earnings, missing fields, and ratio edge cases.

## License

No license has been declared yet. Add one before distributing the project publicly.
