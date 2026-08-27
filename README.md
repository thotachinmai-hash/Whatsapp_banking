# Finacle Banking WhatsApp Assistant

AI-powered WhatsApp banking assistant that accepts voice, text, and document (image/PDF/DOCX) messages, processes them through a LangGraph AI agent and a step-based workflow engine, and responds with real banking information. Built with FastAPI, LangGraph, Groq LLM, Groq Whisper, Groq Vision (OCR), PostgreSQL, Redis, and WhatsApp Business Cloud API.

Every incoming message is first matched against a registered-customer database by WhatsApp number (the "@lid"). Known customers get a personalized greeting and service menu; unknown numbers are guided through an Aadhaar/PAN registration flow before anything else, which also opens their first account. From there, customers can check balances, browse categorized transactions and spend summaries, transfer money, deposit a cheque by photo, apply for a loan, and update KYC — all in natural language over WhatsApp, with every workflow safely cancellable mid-flow.

---

## Features

- **Registration gate** — every message is checked against the `customers` table by phone number.
  - **Registered**: greeted by name with a menu of available services on session start or on an explicit "hi"/"menu"/"help".
  - **Unregistered**: walked through a conversational onboarding flow (name → Aadhaar card image → PAN card image → confirmation → account type) before any banking feature is accessible. Confirming creates both the customer record and their first account.
- **Interruptible workflows** — any active workflow (registration, cheque, loan, KYC, transfer) can be stopped at any step by replying *Cancel*, *Stop*, or a bare greeting like *Hi* that's out of scope for the current step.
  - For **registration**, this returns to the plain welcome/registration prompt only — an unregistered customer has no accounts or services yet, so the transactional service menu is never shown to them.
  - For every other workflow, it cancels cleanly ("nothing was submitted or changed") and shows the full service menu, since the customer is already registered and that menu is meaningful to them.
- **Money transfer** — pick a saved beneficiary or add a new one, choose an amount and source account, then confirm (or edit) before the transfer is executed.
- **Cheque deposit** — upload a photo of a cheque; Groq Vision OCRs the bank, branch, payee, amount, cheque number, and signatory. Missing/invalid mandatory fields (payee, amount) trigger a correction step — re-upload or reply with `Key: value` text. On success, a unique request ID (`CHQ-XXXXXXXX`) is generated and persisted; ask the assistant to check its status any time.
- **Loan application & KYC update** — select a loan type or upload a KYC document, fill in the required fields (by document upload or `Field: value` text), confirm, and get a trackable request ID.
- **Transactions & spend insights** — transactions are tagged with a category (groceries, bills, rent, salary, transport, entertainment, shopping, etc.). Ask for recent transactions filtered by date range/type/category, or a spend summary broken down by category.
- **Voice & document support** — voice notes are transcribed with Groq Whisper; images/PDF/DOCX are parsed with Groq Vision.
- **Correlated logging** — every request gets a short trace ID, threaded through the registration gate, workflow manager, every workflow processor, and every tool call, so a single conversation turn can be followed end-to-end in `logs/app.log`.

---

## Architecture

```mermaid
flowchart TD
    A[WhatsApp User\nSends text, voice, or document] -->|WhatsApp message| B[WhatsApp Business\nCloud API]
    B -->|Webhook POST| C[Public HTTPS URL\nngrok for local dev]
    C -->|Forwards to| D[FastAPI App\nPort 8001]
    D --> E[Message Handler]
    E -->|voice message| F[Groq Whisper\nVoice to Text]
    E -->|document image/PDF/DOCX| DOC[Groq Vision OCR\nDocument Parser]
    E -->|text / transcribed / parsed| G[run_agent]
    F --> G
    DOC --> G
    G --> RG{Registration Gate\ncustomers lookup}
    RG -->|unregistered| ONB[Onboarding Workflow\nname, Aadhaar, PAN\ncreates customer + account]
    RG -->|registered, greeting| MENU[Greeting + Service Menu]
    RG -->|registered, normal message| WF{Active Workflow?}
    WF -->|cheque, loan, KYC,\nor transfer| CHQ[Workflow Processor\nvalidate, correct, persist]
    WF -->|none| H[LangGraph Agent\nGroq LLM tool-calling]
    H -->|tool call| I[Banking Tools\nbalance, transactions,\nspend summary, cheque/loan status]
    I --> J[(PostgreSQL\naccounts, transactions, customers,\ncheque/loan/kyc requests)]
    ONB --> J
    CHQ --> J
    H --> K[(Redis\nSession memory + active\nworkflow state, 1h TTL)]
    RG --> K
    G -->|response text| D
    D -->|send reply via WhatsApp Cloud API| B
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
git clone https://github.com/thotachinmai-hash/Whatsapp_banking.git
cd Whatsapp_banking
cp .env.example .env
```

Edit `.env` and fill in your values:

```env
SARVAM_API_KEY=your_SARVAM_API_KEY_here
SARVAM_STT=saaras:v3
SARVAM_TTS=bulbul:v3

DATABASE_URL=postgresql://banking_user:banking_pass@localhost:5433/banking_db
REDIS_URL=redis://localhost:6380

ACCESS_TOKEN=your_whatsapp_cloud_access_token_here
PHONE_NUMBER_ID=your_whatsapp_phone_number_id_here
VERIFY_TOKEN=your_webhook_verify_token_here

WEBHOOK_SECRET=
```

---

### Step 2 — Start infrastructure

Start PostgreSQL and Redis first:

```bash
docker compose up postgres redis -d
```

---

### Step 3 — Build and start the FastAPI app

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

### Step 4 — Set up a public HTTPS tunnel

WhatsApp Business Cloud API requires a public HTTPS webhook endpoint — local URLs aren't reachable from Meta's servers, so use ngrok for local development.

Install ngrok, then add your auth token from [dashboard.ngrok.com/get-started/your-authtoken](https://dashboard.ngrok.com/get-started/your-authtoken):

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

(Skip this step if deploying to a server with its own public HTTPS address.)

---

### Step 5 — Configure the WhatsApp Business Cloud webhook

In the Facebook Developer console for your WhatsApp Business app, set the webhook callback URL to your public tunnel root:

```text
https://YOUR_NGROK_URL/
```

Use the same `VERIFY_TOKEN` value from your `.env` as the verify token, and subscribe to the `messages` event (plus any other messaging events your app needs).

If configured correctly, Facebook sends a verification request and your app responds with the verify token. The app listens on the root path (`/`) for incoming webhook events.

---

### Step 6 — Test the full flow

Send a WhatsApp message to the registered phone number. Your FastAPI app receives the webhook at `/` and processes it directly.

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
    "phone_number": "XXXXXXXXXXXX",
    "message": "What is my balance for account GB12FNCL00010001234567?"
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
| http://127.0.0.1:4040 | ngrok inspector |

---

## Seed Data

`customers.phone_number` must match `accounts.phone_number` for the same person — the registration gate and "which account is linked to this number" lookups both key off it, so a mismatch makes a real customer look unregistered.

Sample cheque requests seeded for testing cheque status lookup (belong to John Smith / Sarah Johnson's numbers above):

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
What is my balance?
Show me the last 5 transactions
How much did I spend on groceries this month?
I want to deposit a cheque
Check status of CHQ-A1B2C3D4
I want to apply for a loan
Transfer ₹50 to Priya Sharma
```

To test **registration/onboarding**, message from `XXXXXXXXXXXX` (unregistered) and follow the prompts (Aadhaar/PAN are uploaded as document images, not typed):

```
hi
Michael Brown
[upload Aadhaar card image]
[upload PAN card image]
yes
1
```

To test the **interrupt fix**, send `hi` (or `cancel`/`stop`) partway through any of the above flows — registration returns to the plain welcome prompt with no service menu; every other workflow cancels and shows the full menu.

---

## Docker Services

| Container | Image | Port |
|---|---|---|
| whatsapp_postgres | postgres:15 | 5433 |
| whatsapp_redis | redis:7-alpine | 6380 |
| whatsapp_app | whatsapp-banking-app | 8001 |

---

## Database Schema

`infra/postgres/init.sql` creates 7 tables, seeded with test data on first init:

| Table | Purpose | Seeded rows |
|---|---|---|
| `accounts` | Bank accounts (number, holder, balance, currency, type). A customer can have more than one — tools that don't get an explicit account number look these up by phone. | 3 |
| `transactions` | Per-account ledger, tagged with a `category` (groceries, bills, rent, salary, transport, entertainment, shopping, isa, bonus, interest, transfer, other) | 45 (15/account, 3 months) |
| `customers` | Registered WhatsApp numbers → name, Aadhaar (unique), PAN (unique), date of birth, guardian name, address | 2 (John Smith, Sarah Johnson) — `XXXXXXXXXXXX` intentionally left unregistered to test onboarding |
| `cheque_requests` | Cheque deposit requests, keyed by a unique `request_id` (`CHQ-XXXXXXXX`), with status PENDING/COMPLETED/REJECTED | 3 |
| `loan_requests` | Loan applications, keyed by `request_id` (`LOAN-XXXXXXXX`), details stored as JSONB | 0 |
| `kyc_requests` | KYC update submissions, keyed by `request_id` (`KYC-XXXXXXXX`), details stored as JSONB | 0 |
| `sessions` | Legacy per-phone session counter table (unused by current code — session state actually lives in Redis) | 0 |

Verified: all tables and their constraints exist as expected (`customers` has unique constraints on `phone_number`/`aadhaar_number`/`pan_number`; the `*_requests` tables each have a unique constraint on `request_id`), no orphaned transactions (every `transactions.account_id` resolves to a real account), every account's `balance` matches its most recent transaction's `balance_after`, and `customers.phone_number` matches `accounts.phone_number` for the same person.

**Postgres always re-initializes from `init.sql` on every `docker compose down` + `up`.** There is no persistent volume for the Postgres data directory (see `docker-compose.yml`) — this is deliberate, so `init.sql` and the running database can never drift out of sync. A plain restart (`docker compose restart postgres`, or leaving containers running) keeps data; removing and recreating the container (`down` then `up`, with or without `-v`) always wipes it back to exactly what `init.sql` defines. This means any real registrations/accounts/cheques created during a session are test data only and will not survive a `down`/`up` cycle — see [Important Notes](#important-notes).

---

## Project Structure

**app/** — Application code
- `main.py` — FastAPI entry point and webhook receiver
- `agent/agent.py` — LangGraph agent (registration gate → active workflow → tool-calling LLM); `trace_id` flows from here into every layer below
- `agent/tools.py` — Banking tools — balance, transactions, spend summary, cheque status, loan status, start cheque workflow
- `services/whatsapp.py` — WhatsApp Business Cloud API client to send messages and download media
- `services/transcription.py` — Groq Whisper voice to text
- `services/document_parser.py` — Groq Vision OCR for images/PDF/DOCX
- `services/registration_gate.py` — Looks up the sender in `customers`; greets or starts onboarding; owns `GREETING_KEYWORDS`, reused by the workflow manager to detect a mid-workflow interrupt
- `services/menu.py` — Shared service-menu and onboarding-welcome text shown across the app
- `services/receipts.py` — PDF receipt generation for completed cheque/loan/KYC/transfer requests, sent as a WhatsApp document
- `services/message_handler.py` — Routes voice/text/document, runs agent, sends response
- `workflows/manager.py` — Routes a message to the active workflow's processor; owns the cancel/interrupt logic (explicit *Cancel*/*Stop* or a bare greeting) and the workflow-boundary/conversational-question handling that lets customers ask questions without losing their place mid-workflow
- `workflows/memory.py` — Redis-backed workflow state (create/get/update/complete)
- `workflows/constants.py` — Workflow types, statuses, and step constants
- `workflows/processors/onboarding.py` — Name → Aadhaar image → PAN image → confirm → create customer → select account type → open account
- `workflows/processors/cheque.py` — OCR validation, correction loop, cheque request persistence
- `workflows/processors/loan.py` — Loan type selection, form collection (image or text), confirmation, request persistence
- `workflows/processors/kyc.py` — KYC document/field collection, confirmation, request persistence
- `workflows/processors/transfer.py` — Beneficiary selection, amount, source account, confirmation
- `api/routes.py` — REST API endpoints (accounts, customers, cheque requests — for testing/debugging)
- `database.py` — PostgreSQL queries
- `memory.py` — Redis session memory per phone number, active-account cache
- `metrics.py` — Metrics tracking with trace ID
- `logger.py` — Daily rotating logs, console output

**infra/postgres/init.sql** — Database schema and seed data (`accounts`, `transactions`, `sessions`, `customers`, `cheque_requests`, `loan_requests`, `kyc_requests`)

**Root files**
- `docker-compose.yml` — All 3 services (postgres, redis, app)
- `Dockerfile` — Non-root Python container
- `.env.example` — Environment variable template
- `README.md` — This file

---

## Important Notes

**ngrok URL changes on every restart.** Each time you restart ngrok you get a new public URL and must re-register the webhook. To get a permanent URL upgrade to a paid ngrok plan or deploy to a cloud server.

**This project no longer uses OpenWA.** WhatsApp integration is now via the WhatsApp Business Cloud API, and no OpenWA session volume is required.

**Groq rate limits** — llama-3.3-70b-versatile has 100K daily tokens on the free tier. Switch to `qwen/qwen3-32b` in `.env` for 500K daily tokens.

**WEBHOOK_SECRET is optional.** Leave it empty for development. Set a random string in production for webhook authentication.

**The Postgres database is ephemeral by design.** There is no persistent Docker volume for Postgres data — every `docker compose down` + `up` recreates the container from a clean state and re-runs `infra/postgres/init.sql` from scratch. This is intentional: it guarantees the schema/seed data you see always matches exactly what's in `init.sql`, with no possibility of drift between an old running database and newer code. The trade-off is that **any data created during a session — new registrations, auto-opened accounts, cheque deposits — is lost on the next `down`/`up` cycle.** This is fine for local development and testing, but if you ever want this app to hold real, durable customer data, you'll need to reintroduce a named volume on the `postgres` service in `docker-compose.yml`. A plain `docker compose restart` (or simply leaving containers running) does **not** lose data — only removing and recreating the container does.

---

## Known Limitations

- **Groq free-tier daily token limit (100K TPD).** Heavy testing can exhaust this within a session — the app catches `429` responses and replies with "the service is temporarily busy" instead of crashing, but no LLM-driven replies (balance/transactions/spend/cheque-status questions) will work until the quota resets. Registration, the greeting/menu, and every deterministic workflow (cheque, loan, KYC, transfer) are unaffected since they don't call the LLM. Switch to `qwen/qwen3-32b` in `.env` for a 500K daily budget if this is a problem.
- **Onboarding collects Aadhaar/PAN as card images**, not typed text — the vision document parser extracts and format-validates the ID values from the photo.
- **`workflows/memory.py`'s own log lines don't carry a trace ID** (create/get/update/delete workflow state) — every decision made *about* a workflow does (registration gate, workflow manager, every processor), but the low-level Redis read/write lines don't yet. Not a blocker for tracing a conversation, just slightly less granular than the rest.

### Fixed since last review

- **Mid-workflow interrupts didn't work for a bare greeting, and registration's cancel response was wrong.** Only explicit phrases like "cancel"/"stop" interrupted an active workflow — a plain "hi" sent mid-flow just got reinterpreted as if it were the expected input for that step (e.g. rejected as an invalid name). Separately, cancelling *any* workflow — including registration — showed the full transactional service menu (transfer, balance, cheque, etc.), which is meaningless to someone who isn't registered and has no accounts yet. Fixed in `workflows/manager.py`: a bare greeting ("hi", "menu", "help", ...) now interrupts a workflow exactly like "cancel" does, and cancelling registration specifically returns to the plain welcome/registration prompt instead of the service menu. Every other workflow keeps showing the full menu on cancel, since that customer is already registered.
- **Brand renamed from HSBC to Finacle Banking** throughout the app, seed data, and config — including the account-number bank code (`GB..HSBC...` → `GB..FNCL...`), the OTP SMS text, and the OpenWA session name default.
- **Workflow-layer logging had no trace ID and two processors (`transfer.py`, `kyc.py`) had none at all.** `WorkflowManager.handle()`/`start_requested()` and every processor (`onboarding`, `cheque`, `loan`, `kyc`, `transfer`) now accept and log with the same trace ID used everywhere else, and step transitions, validation failures, and request creation/cancellation are now logged consistently across all five workflows — a single conversation turn can be grepped out of `logs/app.log` by its trace ID end-to-end.
- **`customers.phone_number` didn't match `accounts.phone_number`** for the two seeded registered customers, which would make a real customer look unregistered. Fixed in `init.sql`.
