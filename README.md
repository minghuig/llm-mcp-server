# LLM MCP Server

A simple MCP (Model Context Protocol) server that provides tools to query Claude, ChatGPT, and Gemini.

## Features

- `query_claude`: Send messages to Claude (Anthropic API) with multi-turn conversation support
- `query_chatgpt`: Send messages to ChatGPT (OpenAI API) with multi-turn conversation support
- `query_gemini`: Send messages to Gemini (Google API) with multi-turn conversation support
- AI-to-AI conversation context automatically included in system prompts

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -e .
```

3. Create a `.env` file with your API keys:
```bash
cp .env.example .env
# Edit .env and add your actual API keys
```

## Usage

Run the MCP server:
```bash
python server.py
```

## Configuration

Configure your Claude Desktop or other MCP client to connect to this server.

Example configuration for Claude Desktop (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "llm": {
      "command": "python",
      "args": ["/absolute/path/to/server.py"]
    }
  }
}
```

The server will automatically load API keys from the `.env` file. Alternatively, you can override them by adding an `env` section to the config above.

## Available Tools

### query_chatgpt
Send a message to ChatGPT (OpenAI). Supports multi-turn conversations using server-side storage.

Parameters:
- `message` (required): The message to send
- `model` (optional): Model to use (default: "gpt-4.1-2025-04-14")
- `system_prompt` (optional): System prompt for context
- `temperature` (optional): 0.0 to 2.0 (default: 0.7)
- `max_tokens` (optional): Maximum response length
- `previous_response_id` (optional): Response ID from a previous call to continue the conversation

**Multi-turn conversations:**
The tool returns a Response ID in the output. Pass this ID to `previous_response_id` in subsequent calls to continue the conversation. OpenAI stores the conversation history server-side.

Example workflow:
1. First call: `query_chatgpt("Hello!")` → Returns `[Response ID: resp_abc123]`
2. Follow-up: `query_chatgpt("How are you?", previous_response_id="resp_abc123")`

### query_claude
Send a message to Claude (Anthropic). Supports multi-turn conversations using server-side context management.

Parameters:
- `message` (required): The message to send
- `model` (optional): Model to use (default: "claude-sonnet-4-5-20250929")
- `system_prompt` (optional): System prompt for context
- `temperature` (optional): 0.0 to 1.0 (default: 1.0)
- `max_tokens` (optional): Maximum response length (default: 4096)
- `context_id` (optional): Context ID from a previous call to continue the conversation

**Multi-turn conversations:**
The tool returns a Context ID in the output (format: `ctx_claude_...`). Pass this ID to `context_id` in subsequent calls to continue the conversation. The server stores conversation history in memory with automatic limits to prevent memory issues.

Context IDs are provider-specific and cannot be used across different providers (e.g., a Claude context_id cannot be used with Gemini).

Example workflow:
1. First call: `query_claude("Hello!")` → Returns `[Context ID: ctx_claude_abc123def456]`
2. Follow-up: `query_claude("How are you?", context_id="ctx_claude_abc123def456")`

### query_gemini
Send a message to Gemini (Google). Supports multi-turn conversations using server-side context management.

Parameters:
- `message` (required): The message to send
- `model` (optional): Model to use (default: "gemini-2.5-flash")
- `system_prompt` (optional): System prompt for context
- `temperature` (optional): 0.0 to 2.0 (uses model default if not specified)
- `max_tokens` (optional): Maximum response length (uses model default if not specified)
- `context_id` (optional): Context ID from a previous call to continue the conversation

**Multi-turn conversations:**
The tool returns a Context ID in the output (format: `ctx_gemini_...`). Pass this ID to `context_id` in subsequent calls to continue the conversation. The server stores conversation history in memory with automatic limits to prevent memory issues.

Context IDs are provider-specific and cannot be used across different providers (e.g., a Gemini context_id cannot be used with Claude).

Example workflow:
1. First call: `query_gemini("Hello!")` → Returns `[Context ID: ctx_gemini_abc123def456]`
2. Follow-up: `query_gemini("How are you?", context_id="ctx_gemini_abc123def456")`
