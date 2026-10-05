# 🚀 DeepSeek-API: Complete Setup & Deployment Guide

This guide will walk you through setting up and running **DeepSeek-API** on your machine (Windows, macOS, or Linux).

Whether you are connecting it to **OpenCode**, **Cursor**, **Aider**, **Continue.dev**, or writing your own Python scripts, follow these step-by-step instructions.

---

## 📋 Table of Contents
1. [Prerequisites](#1-prerequisites)
2. [Installation & Virtual Environment](#2-installation--virtual-environment)
3. [Authentication (One-Time Login)](#3-authentication-one-time-login)
   - [Method A: Interactive Login (Default)](#method-a-interactive-login-default)
   - [Method B: Reusing Your Existing Browser Profile (Chrome, Opera, Brave, Edge)](#method-b-reusing-your-existing-browser-profile)
4. [Starting the Local Server](#4-starting-the-local-server)
5. [Connecting to AI Coding Agents & IDEs](#5-connecting-to-ai-coding-agents--ides)
   - [OpenCode](#opencode)
   - [Cursor IDE](#cursor-ide)
   - [Aider](#aider)
   - [Continue.dev](#continuedev)
   - [Python OpenAI SDK](#python-openai-sdk)
6. [Available Models & Parameters](#6-available-models--parameters)
7. [Environment Variables Reference](#7-environment-variables-reference)
8. [Troubleshooting & FAQs](#8-troubleshooting--faqs)

---

## 1. Prerequisites

- **Python 3.9 to 3.13** installed on your system.
- A free account on [chat.deepseek.com](https://chat.deepseek.com).
- An internet connection.

---

## 2. Installation & Virtual Environment

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/Deepseek-API.git
cd Deepseek-API
```

### Step 2: Create a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```
*(If PowerShell gives a script permission error, run: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`)*

**On Windows (Command Prompt - CMD):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
playwright install chromium
```

### Step 4: Create Environment File
```bash
# Windows:
copy .env.example .env

# macOS / Linux:
cp .env.example .env
```

---

## 3. Authentication (One-Time Login)

DeepSeek requires solving a browser human-verification check (CAPTCHA). You only need to do this **once**.

### Method A: Interactive Login (Default)
Run the authentication script:
```bash
python -m deepseek.auth
```
1. A Chromium browser window will pop up.
2. Sign in using your Google account, Email, or Phone number.
3. Solve any slider / puzzle verification.
4. Once you land on the DeepSeek chat screen, the terminal will print:
   ```
   [auth] Success! Saved session to session/user.json
   ```
5. You can now close the browser window.

---

### Method B: Reusing Your Existing Browser Profile
If you are already logged into DeepSeek in **Chrome**, **Opera**, **Brave**, or **Edge**, you don't even need to re-login! You can point Playwright directly to your existing browser user profile directory.

Edit your `.env` file and set `DEEPSEEK_PROFILE_DIR`:

#### 🌐 Windows Paths:
- **Opera / Opera GX**:
  ```env
  DEEPSEEK_PROFILE_DIR="C:\Users\<YourUsername>\AppData\Roaming\Opera Software\Opera Stable"
  # For Opera GX:
  # DEEPSEEK_PROFILE_DIR="C:\Users\<YourUsername>\AppData\Roaming\Opera Software\Opera GX Stable"
  ```
- **Google Chrome**:
  ```env
  DEEPSEEK_PROFILE_DIR="C:\Users\<YourUsername>\AppData\Local\Google\Chrome\User Data"
  ```
- **Brave Browser**:
  ```env
  DEEPSEEK_PROFILE_DIR="C:\Users\<YourUsername>\AppData\Local\BraveSoftware\Brave-Browser\User Data"
  ```
- **Microsoft Edge**:
  ```env
  DEEPSEEK_PROFILE_DIR="C:\Users\<YourUsername>\AppData\Local\Microsoft\Edge\User Data"
  ```

#### 🍏 macOS Paths:
- **Google Chrome**:
  ```env
  DEEPSEEK_PROFILE_DIR="/Users/<YourUsername>/Library/Application Support/Google/Chrome"
  ```
- **Opera**:
  ```env
  DEEPSEEK_PROFILE_DIR="/Users/<YourUsername>/Library/Application Support/com.operasoftware.Opera"
  ```

#### 🐧 Linux Paths:
- **Google Chrome**:
  ```env
  DEEPSEEK_PROFILE_DIR="/home/<YourUsername>/.config/google-chrome"
  ```

> **Important**: Close the browser before running `python -m deepseek.auth` so the browser profile database is not locked by the OS.

---

## 4. Starting the Local Server

Run the entry point:
```bash
python app.py
```

The server will start on **`http://localhost:8000`** (or `http://127.0.0.1:8000`).

You can verify it by opening your browser or running:
```bash
curl http://localhost:8000/healthz
```
Expected output:
```json
{
  "status": "ok",
  "service": "deepseek-openai-api",
  "session": {"has_session": true, "has_token": true},
  "pow_queue": {"max_workers": 4, "active_solvers": 0}
}
```

---

## 5. Connecting to AI Coding Agents & IDEs

The server provides a standard OpenAI-compatible API interface.

### OpenCode
In OpenCode provider configuration:
- **Provider**: Custom / OpenAI Compatible
- **Base URL**: `http://localhost:8000/v1` (or `http://localhost:8000`)
- **API Key**: `not-needed` (or any random string)
- **Model**: `deepseek-chat` or `deepseek-reasoner`

### Cursor IDE
1. Open Cursor **Settings** -> **Models**.
2. Under **OpenAI API Key**, toggle **Override OpenAI Base URL**.
3. Set Base URL: `http://localhost:8000/v1`
4. Enter any dummy API key (e.g. `sk-local-test`).
5. Add model names: `deepseek-chat` and `deepseek-reasoner`.

### Aider
Run Aider directly from your command line:
```bash
aider --openai-api-base http://localhost:8000/v1 --openai-api-key not-needed --model deepseek-chat
```
For deep reasoning mode:
```bash
aider --openai-api-base http://localhost:8000/v1 --openai-api-key not-needed --model deepseek-reasoner
```

### Continue.dev
In your `~/.continue/config.json`:
```json
{
  "models": [
    {
      "title": "DeepSeek Chat",
      "provider": "openai",
      "model": "deepseek-chat",
      "apiBase": "http://localhost:8000/v1",
      "apiKey": "not-needed"
    },
    {
      "title": "DeepSeek Reasoner (R1)",
      "provider": "openai",
      "model": "deepseek-reasoner",
      "apiBase": "http://localhost:8000/v1",
      "apiKey": "not-needed"
    }
  ]
}
```

### Python OpenAI SDK
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Explain async queues in Python."}],
    stream=True
)

for chunk in response:
    content = chunk.choices[0].delta.content or ""
    print(content, end="", flush=True)
```

---

## 6. Available Models & Parameters

| Model Name | Description | Thinking Mode (Reasoning) |
| :--- | :--- | :--- |
| `deepseek-chat` | Standard V3 Chat model (Fast & concise) | Optional via `"thinking": true` |
| `deepseek-reasoner` | Reasoning model (R1 CoT stream) | **Always Enabled** (thinking chunks) |
| `deepseek-coder` | Optimized for coding tasks | Optional |
| `gpt-4o` | Alias for agent compatibility | Standard |
| `gpt-3.5-turbo` | Alias for agent compatibility | Standard |

### Extra Parameters (OpenAI `extra_body`):
- `thinking`: `true` | `false` (forces DeepThink reasoning block)
- `search`: `true` | `false` (forces live web search)

---

## 7. Environment Variables Reference

Create or edit your `.env` file with these configuration options:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `PORT` | `8000` | HTTP port for the API server |
| `HOST` | `127.0.0.1` | Network interface (`0.0.0.0` for LAN access) |
| `API_KEY` | *(None)* | Optional custom bearer key to protect your server |
| `RATE_LIMIT_PER_MINUTE` | `60` | Request rate limit per IP |
| `POW_WORKERS` | `4` | Concurrency worker pool size for WASM PoW solver |
| `POW_SOLVE_TIMEOUT` | `25.0` | Max seconds before a PoW challenge solver times out |
| `DEEPSEEK_PROFILE_DIR` | *(None)* | Path to custom browser profile (Chrome/Opera/etc.) |
| `DEBUG_REQUESTS` | `0` | Set `1` to log incoming prompt & tool calls in terminal |

---

## 8. Troubleshooting & FAQs

### Q1: `401 Unauthorized` or `login_required` error?
**Answer**: DeepSeek session tokens periodically expire. The server automatically attempts headless recovery, but if a manual puzzle check is triggered by Cloudflare, simply run:
```bash
python -m deepseek.auth
```
Solve the CAPTCHA once, and the server will resume automatically.

### Q2: `Proof-of-work solver queue busy` (503)?
**Answer**: If your agent fires dozens of requests concurrently, increase `POW_WORKERS` in `.env`:
```env
POW_WORKERS=8
```

### Q3: How to run alongside GLM-API?
**Answer**: DeepSeek-API runs on port `8000` and GLM-API runs on port `8001`. You can run both in separate terminal tabs without any conflict!

### Q4: PowerShell says `Execution of scripts is disabled on this system`?
**Answer**: Run this once in PowerShell:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```
