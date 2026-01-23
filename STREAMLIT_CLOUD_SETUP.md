# Streamlit Cloud Deployment Guide

## Przygotowanie do Deploymentu

### 1. Wymagania
- GitHub repo z kodem
- Streamlit Cloud konto (free tier wystarczy)
- OpenAI lub DeepSeek API key (opcjonalne - Ollama lokalnie działa bez klucza)

### 2. Struktura Plików

```
Financial_report_app/
├── app.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
├── rag_system.py
├── financial_calculator.py
├── data_fetcher.py
├── llm_integrator.py
└── diagnostics.py
```

### 3. Requirements.txt

Plik zawiera:
- `langchain-huggingface` - dla embeddings na Streamlit Cloud
- `sentence-transformers` - model embeddings
- `faiss-cpu` - wektor store

**Ważne**: Używamy `faiss-cpu` zamiast `faiss-gpu` na Streamlit Cloud (brak GPU).

### 4. Streamlit Cloud Cache

**HuggingFace Embeddings Model Caching:**

```
Pierwsze uruchomienie (cold start):
- Pobiera model (all-MiniLM-L6-v2) ~100MB
- Ścieżka: ~/.cache/huggingface/hub/
- Czas: 20-30 sekund
- Wynik: ✅ Model cachowany

Kolejne uruchomienia:
- Model już w cache
- Ładuje się z dysku
- Czas: <1 sekunda
```

**Streamlit Cloud Filesystem:**
- Domyślnie Streamlit Cloud ma `/tmp/` dla cache
- HuggingFace automatycznie cachuje w `~/.cache/`
- Cache **NIE** persystuje między restartami (libre tier)
- Pro/Enterprise: Cache zostaje między restartami

### 5. Deployment Steps

#### Step 1: Git Push
```bash
git add .
git commit -m "Add RAG system with HuggingFace embeddings for Streamlit Cloud"
git push origin main
```

#### Step 2: Streamlit Cloud
1. Idź na https://share.streamlit.io/
2. Kliknij "New app"
3. Wybierz repo: `mik-myrepo`
4. Branch: `main`
5. File path: `Financial_report_app/app.py`
6. Kliknij "Deploy"

#### Step 3: Czekaj na Deploy
- Streamlit Cloud instaluje `requirements.txt`
- Przy pierwszym loadzie strony model się ściąga
- Wyświetli się komunikat "⏳ Loading embeddings..."
- Po ~20-30s aplikacja będzie gotowa

### 6. Environment Variables na Streamlit Cloud

Jeśli chcesz przechowywać API keys bezpiecznie:

1. Idź do ustawień app (⚙️ Settings)
2. Secrets (tab)
3. Dodaj sekrety w TOML:
```toml
openai_api_key = "sk-..."
deepseek_api_key = "sk-..."
```

Dostęp w app.py:
```python
import streamlit as st
openai_key = st.secrets.get("openai_api_key")
```

### 7. Troubleshooting

#### Problem: "Module not found: langchain_huggingface"
**Rozwiązanie:** 
- Sprawdź czy `requirements.txt` ma `langchain-huggingface>=0.0.1`
- Poczekaj aż Streamlit Cloud zainstaluje pakiet (~2 min)

#### Problem: "HuggingFace model download timeout"
**Rozwiązanie:**
- Normalne przy pierwszym runnie na Streamlit Cloud
- Model pobiera się z huggingface.co (~20-30s)
- Poczekaj lub przeładuj stronę
- Kolejne runs będą szybkie

#### Problem: "DeepSeek RAG nie działa"
**Rozwiązanie:**
- DeepSeek API key wpisany? ✓
- Używamy HuggingFace embeddings (nie OpenAI) ✓
- To powinno działać teraz

#### Problem: "Out of memory" error
**Rozwiązanie:**
- Streamlit Cloud free tier: ~512MB RAM
- FAISS + HuggingFace model: ~200MB
- Zostaje ~300MB dla danych
- Jeśli crash: upgrade do Streamlit Cloud Pro

### 8. Performance Notes

**Typical Load Times:**
```
Cold Start (first deployment):
- requirements.txt install: 2-3 min
- First page load: 20-30s (model download)
- Total: ~3-4 min

Warm Start (subsequent runs):
- Page load: <1s
- RAG query: 1-3s (depends on LLM provider)
- Chart rendering: <1s
```

**Optimization Tips:**
1. **Czyszczenie Secrets**: Ulož API keys w Streamlit Secrets
2. **Zmiana Modelu Embeddings**: Jeśli chcesz szybciej - użyj mniejszego modelu w `rag_system.py`:
   ```python
   "sentence-transformers/all-MiniLM-L6-v2"  # Current (dobry balans)
   "all-MiniLM-L6-v2"  # Mniejszy - szybciej
   ```

3. **Cache Chart Renderings**: Streamlit robi to automatycznie
4. **Избегай Large Files**: Nie commituj CSV/pickle > 100MB

### 9. Example Deployment

```
GitHub: MikPe1/mik-myrepo
Branch: main
App File: Financial_report_app/app.py

Secrets (optional):
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...

URL: https://share.streamlit.io/mikpe1/mik-myrepo/main/Financial_report_app/app.py
```

### 10. Co się będzie działo gdy zahosujesz

**Użytkownik odwiedzam Streamlit Cloud app:**

1. **Landing** (1-3s):
   - App się startuje
   - Streamlit ładuje `app.py`
   - Initialize session state

2. **API Configuration** (immediate):
   - User wybiera provider (OpenAI/DeepSeek/Ollama)
   - User wpisuje API key

3. **First Load** (20-30s jeśli cold start):
   - User wpisuje ticker i naciska Enter
   - App przygotowuje RAG system
   - HuggingFace model się ściąga z huggingface.co (~100MB)
   - "⏳ Initializing RAG system..." message
   - Po pobraniu model cachuje się (Linux temp directory)

4. **Data Loading** (2-5s):
   - YahooFinance API query dla tickera
   - Financial metrics calculation
   - Chart data preparation

5. **RAG & Analysis** (3-10s):
   - RAG builds knowledge base (5 documents)
   - LLM generates initial summary
   - Chat ready for user queries

6. **Subsequent Runs** (<1s + LLM time):
   - Model już cachowany
   - Tylko data fetch + LLM inference

**Total First Run**: ~40-60s (mostly HuggingFace model download)
**Subsequent Runs**: ~5-15s (just data + LLM)

---

**Notes:**
- Free tier Streamlit Cloud: app sleeps po 1h inactivity
- Cache NIE persystuje między sleep cycles - znów będzie cold start
- Pro/Enterprise: cache stays, faster boots
