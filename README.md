# Tamil Nadu Agriculture Schemes RAG

A Streamlit Retrieval-Augmented Generation application for exploring Tamil Nadu
Government Agriculture - Farmers Welfare Department schemes. The app scrapes the
official schemes page, stores structured data locally, creates a FAISS vector
index with OpenAI embeddings, and answers user questions with source links.

## Main Features

- Beautiful Streamlit chat interface with source expanders and example prompts.
- Polite scraper with redirect and homepage-content detection.
- Local JSON and CSV storage for scraped scheme records.
- LangChain document creation, chunking, FAISS indexing, and retrieval.
- Grounded answers using only retrieved scheme content.
- Sidebar controls for refresh, index rebuild, chat clearing, and retriever size.
- Optional LangSmith tracing for scraping, indexing, retrieval, and answer runs.

## Architecture

```text
app.py          Streamlit interface and user workflow
config.py       Environment loading and validation
scraper.py      Tamil Nadu scheme scraping and local persistence
rag_pipeline.py LangChain document, FAISS, retrieval, and answer logic
data/           Local scraped JSON and CSV files
faiss_index/    Local FAISS index and metadata
```

## RAG Workflow

1. Scraped schemes are converted into `langchain_core.documents.Document`
   objects.
2. Long documents are split with `RecursiveCharacterTextSplitter`.
3. OpenAI embeddings are generated with the configured embedding model.
4. Chunks are saved into a local FAISS index.
5. User questions retrieve the most relevant chunks.
6. The chat model answers using a strict grounding prompt and includes sources.

## Web Scraping Workflow

The scraper first visits `https://www.tn.gov.in/schemes.php` to establish a
session, then loads the Agriculture scheme list page:

```text
https://www.tn.gov.in/scheme_list.php?dep_id=Mg==
```

It follows only approved Tamil Nadu Government links, removes navigation and
footer text, deduplicates records, and refuses to overwrite valid local data
with empty or homepage-like content.

## LangSmith Tracing

Set `LANGSMITH_TRACING=true` and provide `LANGSMITH_API_KEY` to trace LangChain
model and retrieval calls. Tracing metadata includes safe details such as model
names, retriever size, chunk count, and dataset hash. API keys and cookies are
never sent as metadata.

## Directory Structure

```text
TN_Agriculture_Schemes_RAG/
├── app.py
├── scraper.py
├── rag_pipeline.py
├── config.py
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── README.md
├── .streamlit/
│   └── config.toml
├── data/
│   ├── schemes.json
│   └── schemes.csv
├── faiss_index/
│   ├── index.faiss
│   ├── index.pkl
│   └── index_metadata.json
├── docs/
│   ├── TN_Agriculture_Schemes_RAG_Development_Plan.pdf
│   └── TN_Agriculture_Schemes_RAG_Workflow.pdf
└── test/
    ├── test_api.py
    └── test_ui.py
```

`data/` and `faiss_index/` are created automatically when the app starts.
`requirements.txt` holds both runtime and (clearly marked) dev/test dependencies.

## Prerequisites

- Python 3.10 or newer recommended.
- OpenAI API key.
- Optional LangSmith API key.

## Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

## Environment Configuration

Copy `.env.example` to `.env` and fill in your values:

```dotenv
OPENAI_API_KEY=your_real_openai_api_key
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_MAX_OUTPUT_TOKENS=700
OPENAI_TIMEOUT=60

LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=tn-agriculture-schemes-rag
LANGSMITH_ENDPOINT=https://api.smith.langchain.com

SOURCE_URL=https://www.tn.gov.in/scheme_list.php?dep_id=Mg==
SCHEMES_LANDING_URL=https://www.tn.gov.in/schemes.php

CHUNK_SIZE=1000
CHUNK_OVERLAP=150
RETRIEVER_K=4
MAX_INPUT_CHARS=1200
MAX_HISTORY_MESSAGES=50

REQUEST_TIMEOUT=30
REQUEST_RETRIES=3
REQUEST_DELAY_SECONDS=1

DATA_DIRECTORY=data
FAISS_DIRECTORY=faiss_index

APP_ENV=development
ADMIN_ACTIONS_ENABLED=true
ADMIN_PASSWORD=
```

If LangSmith tracing is enabled, add a valid `LANGSMITH_API_KEY`.

### Key variables

- `OPENAI_CHAT_MODEL` — chat model used for grounded answers (verify the exact
  string against the current OpenAI model list before deploying).
- `OPENAI_MAX_OUTPUT_TOKENS` / `OPENAI_TIMEOUT` — cap answer length and request
  time to control cost and latency.
- `MAX_INPUT_CHARS` — maximum question length accepted from the user (also
  reduces prompt-injection surface). `MAX_HISTORY_MESSAGES` bounds chat history.
- `APP_ENV` — set to `production` to hide internal error details in the UI.
- `ADMIN_ACTIONS_ENABLED` / `ADMIN_PASSWORD` — see **Administrative Controls**.

## Administrative Controls

The data-changing sidebar actions — **Refresh Website Data** and **Rebuild FAISS
Index** — are protected, because they trigger scraping and paid embedding calls
and can replace local data:

- `ADMIN_ACTIONS_ENABLED=false` disables these controls entirely.
- `ADMIN_ACTIONS_ENABLED=true` with an empty `ADMIN_PASSWORD` allows them (local
  development convenience).
- `ADMIN_ACTIONS_ENABLED=true` with a set `ADMIN_PASSWORD` requires the password
  to be entered in the sidebar before the controls unlock.

In production (`APP_ENV=production`) configuration validation **requires** a
non-empty `ADMIN_PASSWORD` whenever admin actions are enabled. The password is
read only from the environment and is never stored in source.

## Running the App

```bash
streamlit run app.py
```

Then use the sidebar:

1. Click **Refresh Website Data** to scrape schemes.
2. Click **Rebuild FAISS Index** to create embeddings and the vector index.
3. Ask questions in the Assistant tab.

## Refreshing Scraped Data

Use **Refresh Website Data** in the sidebar. If local data already exists, the
UI asks for confirmation before a successful scrape can replace it. Failed
refresh attempts do not overwrite the last valid JSON or CSV files.

## Rebuilding the FAISS Index

Use **Rebuild FAISS Index** after scraping data or changing the embedding model,
chunk size, or chunk overlap. The app reuses an existing index when metadata
matches the current dataset and settings.

## Example Questions

- What training schemes are available for farmers?
- Are there any schemes related to seed production?
- Which schemes provide subsidies?
- What are the eligibility conditions for the available schemes?
- How can farmers apply for these schemes?

## Data and Index Storage

- `data/schemes.json` stores full structured records.
- `data/schemes.csv` supports spreadsheet review and download.
- `faiss_index/index.faiss` and `faiss_index/index.pkl` store the vector index.
- `faiss_index/index_metadata.json` stores dataset hash, embedding model, chunk
  settings, creation time, scheme count, document count, and chunk count.

## Testing

Install dependencies (`requirements.txt` includes the dev/test tools), then run:

```bash
pip install -r requirements.txt

# Backend / API tests (mocked OpenAI, LangSmith, and government requests)
pytest test/test_api.py -v

# UI tests (Playwright; install browsers once)
python -m playwright install
pytest test/test_ui.py -v --browser chromium
```

Tests never make real paid API calls or real government requests. Markers
(`api`, `ui`, `security`, `smoke`, etc.) are registered in `pyproject.toml`.

## Troubleshooting

- **Missing OpenAI key**: Add `OPENAI_API_KEY` to `.env`.
- **LangSmith warning**: Set `LANGSMITH_TRACING=false` or add
  `LANGSMITH_API_KEY`.
- **No data available**: Use **Refresh Website Data**.
- **Chat disabled**: Use **Rebuild FAISS Index** after valid data exists.
- **Admin buttons disabled**: Set `ADMIN_ACTIONS_ENABLED=true` and enter
  `ADMIN_PASSWORD` (if configured) in the sidebar.
- **FAISS install issues on Windows**: Upgrade pip, then reinstall:

```powershell
python -m pip install --upgrade pip
pip install faiss-cpu
```

## Known Website Redirect Issue

The Tamil Nadu Government page may occasionally redirect automated requests to
the homepage or change its HTML structure. The scraper detects redirects and
homepage markers such as tourism, documents, press releases, forms, and visitor
count. It reports the issue clearly and preserves the last valid local dataset
instead of indexing unrelated homepage content.

## Ethical Scraping Notes

This app scrapes only public government pages, uses a transparent User-Agent,
adds delays between detail-page requests, and does not bypass authentication,
CAPTCHA, robots restrictions, or security controls.

## Security Guidance

- Do not commit real `.env` files.
- Do not place API keys in Streamlit widgets or source text.
- Only load FAISS indexes generated by this local app. FAISS metadata uses local
  pickle-backed storage, so loading untrusted indexes is unsafe.
- The CSV export neutralizes spreadsheet formula injection: values beginning
  with `=`, `+`, `-`, or `@` are prefixed with `'` so Excel/LibreOffice treats
  them as text. The primary JSON keeps the exact source value.
- Data-changing actions are gated by `ADMIN_ACTIONS_ENABLED` / `ADMIN_PASSWORD`
  (see **Administrative Controls**).
- User input is length-bounded (`MAX_INPUT_CHARS`) and the system prompt treats
  retrieved page text as untrusted data, mitigating prompt injection.
- With `APP_ENV=production`, internal error details are hidden from the UI.

## Limitations

- The scraper depends on public website availability and page structure.
- Missing fields are stored as empty strings rather than inferred.
- Answers are limited to retrieved scraped content.
- Users should verify critical or time-sensitive information on the official
  Tamil Nadu Government webpage.

## Disclaimer

This project is an educational assistant and is not an official Tamil Nadu
Government service. Always verify scheme eligibility, benefits, dates,
application procedures, and contact information on official government pages.
