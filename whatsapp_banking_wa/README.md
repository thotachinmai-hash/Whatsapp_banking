# Finacle WhatsApp Banking Assistant

AI-powered WhatsApp banking assistant that accepts voice, text, and document (image/PDF/DOCX) messages, processes them through a LangGraph AI agent and a step-based workflow engine, and responds with real banking information. Built with FastAPI, LangGraph, Groq LLM, Groq Whisper, Groq Vision (OCR), PostgreSQL, Redis, and OpenWA.

Every incoming message is first matched against a registered-customer database by WhatsApp number (the "@lid"). Known customers get a personalized greeting and service menu; unknown numbers are guided through an Aadhaar/PAN registration flow before anything else. From there, customers can check balances, browse categorized transactions and spend summaries, deposit a cheque by photo (with OCR field validation and a trackable request ID), and check the status of a submitted cheque — all in natural language over WhatsApp.

---

## Features

- **Registration gate** — every message is checked against the `customers` table by phone number.
  - **Registered**: greeted by name with a menu of available services on session start or on an explicit "hi"/"menu"/"help".
  - **Unregistered**: walked through a conversational onboarding flow (full name → Aadhaar number → PAN number → confirmation) before any banking feature is accessible.
- **Cheque deposit workflow** — upload a photo of a cheque; Groq Vision OCRs the bank, branch, payee, amount, cheque number, and signatory. Missing mandatory fields (payee, amount) trigger a correction step — re-upload or reply with `Key: value` text. On success, a unique request ID (`CHQ-XXXXXXXX`) is generated and persisted; ask the assistant to "check status of CHQ-XXXXXXXX" any time.
- **Transactions & spend insights** — transactions are tagged with a category (groceries, bills, rent, salary, transport, entertainment, shopping, etc.). Ask for recent transactions filtered by date range/type/category, or a spend summary broken down by category.
- **Voice & document support** — voice notes are transcribed with Groq Whisper; images/PDF/DOCX are parsed with Groq Vision.

---

## Architecture

```mermaid
flowchart TD
    A[WhatsApp User\nSends text, voice, or document] -->|WhatsApp message| B[OpenWA Gateway\nPort 2785]
    B -->|Webhook POST| C[ngrok Public HTTPS URL]
    C -->|Forwards to| D[FastAPI App\nPort 8001]
    D --> E[Message Handler]
    E -->|voice message| F[Groq Whisper\nVoice to Text]
    E -->|document image/PDF/DOCX| DOC[Groq Vision OCR\nDocument Parser]
    E -->|text / transcribed / parsed| G[run_agent]
    F --> G
    DOC --> G
    G --> RG{Registration Gate\ncustomers lookup}
    RG -->|unregistered| ONB[Onboarding Workflow\nname, Aadhaar, PAN]
    RG -->|registered, greeting| MENU[Greeting + Service Menu]
    RG -->|registered, normal message| WF{Active Workflow?}
    WF -->|cheque deposit| CHQ[Cheque Workflow\nvalidate, correct, persist]
    WF -->|none| H[LangGraph Agent\nGroq LLM tool-calling]
    H -->|tool call| I[Banking Tools\nbalance, transactions,\nspend summary, cheque status]
    I --> J[(PostgreSQL\naccounts, transactions,\ncustomers, cheque_requests)]
    ONB --> J
    CHQ --> J
    H --> K[(Redis\nSession memory + active\nworkflow state, 1h TTL)]
    RG --> K
    G -->|response text| D
    D -->|send reply via OpenWA API| B
    B -->|WhatsApp reply| A
    D --> L[Trace ID Logging\nDaily rotating logs\n7 day retention]
    D --> M[Metrics Endpoint\nGET /metrics]
```

---

## Prerequisites

- Docker and Docker Compose
- Groq API key — free at [console.groq.com](https://console.groq.com)
- ngrok account — free at [dashboard.ngrok.com](https://dashboard.ngrok.com)
- A phone with WhatsApp installed
- A second WhatsApp number to receive replies

---

## Setup — Step by Step

### Step 1 — Clone and configure

```bash
git clone https://github.com/dinnyhub/whatsapp-banking-agent.git
cd whatsapp-banking-agent
cp .env.example .env
```

Edit `.env` and fill in your values:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_WHISPER_MODEL=whisper-large-v3-turbo

DATABASE_URL=postgresql://banking_user:banking_pass@localhost:5433/banking_db
REDIS_URL=redis://localhost:6380

OPENWA_URL=http://localhost:2785
OPENWA_API_KEY=get_this_after_step_2
OPENWA_SESSION_ID=get_this_after_step_3

WEBHOOK_SECRET=
```

---

### Step 2 — Start infrastructure and get OpenWA API key

Start PostgreSQL, Redis and OpenWA first:

```bash
docker compose up postgres redis openwa -d
```

Wait 30 seconds then get the auto-generated API key:

```bash
docker exec whatsapp_openwa cat /app/data/.api-key
```

Copy the key and update `.env`:

```env
OPENWA_API_KEY=owa_k1_xxxxxxxxxxxxxxxxxxxx
```

---

### Step 3 — Create WhatsApp session

Open the OpenWA dashboard at **http://localhost:2785**

Enter the API key from Step 2 to login.

Then:
1. Click **Sessions** in the left menu
2. Click **New Session**
3. Name it: `hsbc-assistant`
4. Click **Create**
5. Click **Start** on the session
6. Scan the QR code with WhatsApp on your phone — WhatsApp → Linked Devices → Link a Device
7. Wait for status to show **Connected**

Copy the Session ID shown in the dashboard and update `.env`:

```env
OPENWA_SESSION_ID=a02d3dd1-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

---

### Step 4 — Build and start the FastAPI app

```bash
docker compose build app
docker compose up app -d
```

Verify it is running:

```bash
curl http://localhost:8001/health
```

Expected response:

```json
{
  "status": "healthy",
  "components": {
    "api": "healthy",
    "redis": "connected",
    "postgres": "connected"
  }
}
```

---

### Step 5 — Set up ngrok tunnel

OpenWA requires a public HTTPS URL for webhooks — local URLs are blocked by SSRF protection.

Install ngrok:

```bash
brew install ngrok
```

Add your auth token from [dashboard.ngrok.com/get-started/your-authtoken](https://dashboard.ngrok.com/get-started/your-authtoken):

```bash
ngrok config add-authtoken YOUR_NGROK_TOKEN
```

Start the tunnel in a separate terminal — keep it running:

```bash
ngrok http 8001
```

Copy the public URL shown — looks like:

```
https://2cb6-5-151-181-20.ngrok-free.app
```

---

### Step 6 — Register the webhook

Replace the values with your actual OpenWA API key, Session ID and ngrok URL:

```bash
curl -X POST http://localhost:2785/api/sessions/YOUR_SESSION_ID/webhooks \
  -H "X-API-Key: YOUR_OPENWA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://YOUR_NGROK_URL/webhook/whatsapp",
    "events": ["message.received"]
  }'
```

Expected response:

```json
{
  "id": "ed897686-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "active": true,
  "url": "https://YOUR_NGROK_URL/webhook/whatsapp",
  "events": ["message.received"]
}
```

---

### Step 7 — Test the full flow

Send a WhatsApp message to the linked number:

```
What is my balance for account GB12HSBC00010001234567?
```

You should receive a reply with the account balance.

Check the logs to see the full trace:

```bash
docker logs whatsapp_app --tail=50
```

---

## Test Without WhatsApp

Test the agent directly without going through WhatsApp using Swagger UI at **http://localhost:8001/docs**

Or via curl:

```bash
curl -X POST http://localhost:8001/api/test/message \
  -H "Content-Type: application/json" \
  -d '{
    "phone_number": "447812345678",
    "message": "What is my balance for account GB12HSBC00010001234567?"
  }'
```

---

## URLs

| URL | Description |
|---|---|
| http://localhost:8001 | FastAPI application |
| http://localhost:8001/docs | Swagger UI |
| http://localhost:8001/health | Health check |
| http://localhost:8001/metrics | System metrics |
| http://localhost:2785 | OpenWA dashboard |
| http://127.0.0.1:4040 | ngrok inspector |

---

## Test Accounts

| Account Number | Holder | Balance | Phone Number | Registered? |
|---|---|---|---|---|
| GB12HSBC00010001234567 | John Smith | £2,543.67 | 447812345678 | ✅ Yes |
| GB12HSBC00010007654321 | Sarah Johnson | £15,750.00 | 447987654321 | ✅ Yes |
| GB12HSBC00010009876543 | Michael Brown | £892.34 | 447123456789 | ❌ No — use this number to test onboarding |

Sample cheque requests seeded for testing `check_cheque_status` (all belong to John Smith / Sarah Johnson's numbers above):

| Request ID | Status |
|---|---|
| CHQ-A1B2C3D4 | COMPLETED |
| CHQ-E5F6G7H8 | PENDING |
| CHQ-J9K1L2M3 | REJECTED |

---

## Example Queries

Send these as WhatsApp messages or test via Swagger:

```
hi
What is my balance for account GB12HSBC00010001234567?
Show me the last 5 transactions for account GB12HSBC00010007654321
How much did I spend on groceries this month for account GB12HSBC00010001234567?
I want to deposit a cheque
Check status of CHQ-A1B2C3D4
```

To test **registration/onboarding**, message from `447123456789` (unregistered) and follow the prompts:

```
hi
Michael Brown
345678901299
CDEFG5678H
yes
```

---

## Docker Services

| Container | Image | Port |
|---|---|---|
| whatsapp_postgres | postgres:15 | 5433 |
| whatsapp_redis | redis:7-alpine | 6380 |
| whatsapp_openwa | ghcr.io/rmyndharis/openwa:latest | 2785 |
| whatsapp_app | whatsapp-banking-app | 8001 |

---

## Database Schema

`infra/postgres/init.sql` creates 5 tables, seeded with test data on first init:

| Table | Purpose | Seeded rows |
|---|---|---|
| `accounts` | Bank accounts (number, holder, balance, currency, type) | 3 |
| `transactions` | Per-account ledger, tagged with a `category` (groceries, bills, rent, salary, transport, entertainment, shopping, isa, bonus, interest, transfer, other) | 45 (15/account, 3 months) |
| `customers` | Registered WhatsApp numbers → name, Aadhaar (unique), PAN (unique) | 2 (John Smith, Sarah Johnson) — Michael Brown intentionally left unregistered to test onboarding |
| `cheque_requests` | Cheque deposit requests, keyed by a unique `request_id` (`CHQ-XXXXXXXX`), with status PENDING/COMPLETED/REJECTED | 3 |
| `sessions` | Legacy per-phone session counter table (unused by current code — session state actually lives in Redis) | 0 |

Verified: all 5 tables and their constraints exist as expected (`customers` has unique constraints on `phone_number`/`aadhaar_number`/`pan_number`; `cheque_requests` has a unique constraint on `request_id`), no orphaned transactions (every `transactions.account_id` resolves to a real account), and every account's `balance` matches its most recent transaction's `balance_after`.

Since `init.sql` only runs against an empty Postgres volume, re-seed after schema/data changes with:
```bash
docker compose down -v && docker compose up -d postgres redis
```

---

## Project Structure

**app/** — Application code
- `main.py` — FastAPI entry point and webhook receiver
- `agent/agent.py` — LangGraph agent (registration gate → active workflow → tool-calling LLM)
- `agent/tools.py` — Banking tools — balance, transactions, spend summary, cheque status, start cheque workflow
- `services/whatsapp.py` — OpenWA client to send messages
- `services/transcription.py` — Groq Whisper voice to text
- `services/document_parser.py` — Groq Vision OCR for images/PDF/DOCX
- `services/registration_gate.py` — Looks up the sender in `customers`; greets or starts onboarding
- `services/menu.py` — Shared service-menu text shown to registered customers
- `services/message_handler.py` — Routes voice/text/document, runs agent, sends response
- `workflows/manager.py` — Routes a message to the active workflow's processor, if any
- `workflows/memory.py` — Redis-backed workflow state (create/get/update/complete)
- `workflows/constants.py` — Workflow types, statuses, and step constants
- `workflows/processors/onboarding.py` — Name → Aadhaar → PAN → confirm → create customer
- `workflows/processors/cheque.py` — OCR validation, correction loop, cheque request persistence
- `workflows/processors/kyc.py`, `workflows/processors/loan.py` — Stubs, not yet implemented
- `api/routes.py` — REST API endpoints (accounts, customers, cheque requests — for testing/debugging)
- `database.py` — PostgreSQL queries
- `memory.py` — Redis session memory per phone number
- `metrics.py` — Metrics tracking with trace ID
- `logger.py` — Daily rotating logs

**infra/postgres/init.sql** — Database schema and seed data (`accounts`, `transactions`, `sessions`, `customers`, `cheque_requests`)

**Root files**
- `docker-compose.yml` — All 4 services
- `Dockerfile` — Non-root Python container
- `.env.example` — Environment variable template
- `README.md` — This file

---

## Important Notes

**ngrok URL changes on every restart.** Each time you restart ngrok you get a new public URL and must re-register the webhook. To get a permanent URL upgrade to a paid ngrok plan or deploy to a cloud server.

**OpenWA session persists** in the `whatsapp_openwa_data` Docker volume. If you delete the volume you need to scan the QR code again.

**Groq rate limits** — llama-3.3-70b-versatile has 100K daily tokens on the free tier. Switch to `qwen/qwen3-32b` in `.env` for 500K daily tokens.

**WEBHOOK_SECRET is optional.** Leave it empty for development. Set a random string in production for webhook authentication.

**Resetting the database.** `infra/postgres/init.sql` only runs against an empty Postgres data volume — editing it has no effect on an already-initialized database. To pick up schema/seed changes: `docker compose down -v && docker compose up -d postgres redis`. This wipes all local data (test data only, safe to recreate).

---

## Known Limitations

- **Groq free-tier daily token limit (100K TPD).** Heavy testing can exhaust this within a session — the app catches `429` responses and replies with "the service is temporarily busy" instead of crashing, but no LLM-driven replies (balance/transactions/spend/cheque-status questions) will work until the quota resets. Registration, greeting/menu, and the cheque upload workflow are unaffected since they don't call the LLM. Switch to `qwen/qwen3-32b` in `.env` for a 500K daily budget if this is a problem.
- **Loan and KYC workflows** are wired into the menu with document extraction, mandatory-field validation, confirmation, request persistence, and request IDs.
- **Onboarding collects Aadhaar/PAN as card images** — the existing vision document parser extracts the ID values, which are then format-validated before registration.

### Fixed since last review

- **Multi-turn tool-calling recursion bug (fixed).** `AgentState.messages` in `agent.py` was declared as a plain `list` with no LangGraph reducer, so every graph node's return *replaced* the message history instead of appending to it — the agent lost the original question and prior tool-call context on every hop. In multi-turn conversations (e.g. "hi" → "what's my balance") this reliably caused the agent to loop calling the same tool until it hit the recursion limit and returned a generic error. Fixed by annotating the field with LangGraph's `add_messages` reducer (`Annotated[list, add_messages]`), the standard pattern for LangGraph chat agents. Verified with 6 consecutive multi-turn tool-calling conversations (balance, transactions, spend summary, cheque status) — all succeeded — before the fix's testing exhausted the Groq daily quota.
