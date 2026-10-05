# DeepSeek-API: Rock-Solid OpenAI-Compatible API for DeepSeek (V3 & R1)

A production-grade, thread-safe, OpenAI-compatible proxy server for **DeepSeek (Chat V3 & Reasoner R1)**.

Designed specifically for AI coding agents (**OpenCode**, **Cursor**, **Aider**, **Continue.dev**, **Cline**) and IDE tools.

Runs on **port `8000`** and can run side-by-side with **`GLM-API`** (`8001`)!

> 📖 **Looking for step-by-step setup instructions?** Check out the [Comprehensive Setup & IDE Integration Guide](SETUP_GUIDE.md).

---

## 🌟 Key Features

1. **Free Web Session Automation (`chat.deepseek.com`)**:
   - Uses your personal account via browser session & Playwright.
   - Persistent cookies, auto-login, and automated headless background 401 session recovery.

2. **Full OpenAI Compatibility**:
   - Both `/v1/chat/completions` and `/chat/completions` routes.
   - Both `/v1/models` and `/models` discovery endpoints.
   - Fallback `/v1/embeddings` stub so IDE tools never crash.
   - Diagnostics & health metrics at `/healthz` and `/`.

3. **Tool Calling & Multi-Turn Fidelity**:
   - Automatic OpenAI tool contract injection.
   - Multi-format parser extracting JSON markdown fences, bare JSON, XML tags, ReAct format (`Action:`/`Action Input:`), and function call syntax.
   - Automatic repair for truncated or streaming JSON.
   - Full multi-turn conversation preservation (replays past `tool_calls` and tool execution outputs).

4. **DeepThink Reasoning Stream (R1)**:
   - Real-time streaming of thinking tokens as `delta.reasoning_content` for `deepseek-reasoner`.

5. **Non-Blocking Proof-of-Work (PoW) Queue**:
   - Shared WASM solver engine with bounded worker concurrency (`POW_WORKERS=4`, `POW_SOLVE_TIMEOUT=25s`).
   - Prevents IDE agent deadlocks under rapid concurrent calls.

6. **Browser Profile Reusability (Chrome, Opera, Brave, Edge)**:
   - Reuse your existing signed-in browser profile without logging in every time.

---

## 🚀 Quick Start

### 1. Requirements & Setup

```bash
git clone https://github.com/kuchbhi03834/Deepseek-API.git
cd Deepseek-API
pip install -r requirements.txt
playwright install chromium
```

Copy the example environment file:
```bash
# Windows:
copy .env.example .env

# macOS / Linux:
cp .env.example .env
```

### 2. Authentication (One-Time Login)

```bash
python -m deepseek.auth
```
A Chromium browser window will open. Log into your DeepSeek account and solve the puzzle verification once. The session is automatically saved to `session/user.json`.

*(Note: If you are already logged in on Opera, Chrome, or Edge, see [SETUP_GUIDE.md](SETUP_GUIDE.md) to reuse your browser profile directly).*

### 3. Start the Server

```bash
python app.py
```
Server starts on: `http://localhost:8000`

---

## 🛠️ Configuration & Models

### Available Models

| Model ID | Target Backend | Description | Thinking Tokens |
| :--- | :--- | :--- | :--- |
| `deepseek-chat` | DeepSeek-V3 | Fast flagship chat model | Optional via `"thinking": true` |
| `deepseek-reasoner` | DeepSeek-R1 | Advanced reasoning model | **Always Enabled** (thinking chunks) |
| `deepseek-coder` | DeepSeek-V3 | Optimized for code tasks | Optional |
| `gpt-4o` | DeepSeek-V3 | Alias for agent compatibility | Standard |
| `gpt-3.5-turbo` | DeepSeek-V3 | Alias for agent compatibility | Standard |

---

## 🤖 Connecting to OpenCode / Cursor / Aider

### OpenCode Configuration
In OpenCode settings or provider configuration:
- **Base URL**: `http://localhost:8000/v1`
- **API Key**: `not-needed`
- **Model**: `deepseek-chat` or `deepseek-reasoner`

### Cursor IDE
1. Open Cursor **Settings** -> **Models**.
2. Under **OpenAI API Key**, toggle **Override OpenAI Base URL**.
3. Set Base URL: `http://localhost:8000/v1`
4. Enter any dummy API key (`sk-local-test`).
5. Add model names: `deepseek-chat` and `deepseek-reasoner`.

### Python SDK Example
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Write a python script to sort a dictionary by value."}],
)
print(response.choices[0].message.content)
```

---

## 🧪 Running Tests

A complete automated unit and stress test suite is included:

```bash
py -3.13 -m unittest discover tests
```
All 27 test cases verify tool calling, multi-turn loops, streaming, error resilience, and PoW queue concurrency.

---

## 📄 License

MIT License.
