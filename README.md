# LLM MCP Server

A simple MCP (Model Context Protocol) server that provides tools to query Claude, ChatGPT, and Gemini.

## Features

- `query_claude`: Send messages to Claude (Anthropic API) with multi-turn conversation support
- `query_chatgpt`: Send messages to ChatGPT (OpenAI API) with multi-turn conversation support
- `query_gemini`: Send messages to Gemini (Google API) with multi-turn conversation support
- `export_conversation`: Export conversations to markdown files
- AI-to-AI conversation context automatically included in system prompts
- System prompt persistence across conversations

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
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
Send a message to ChatGPT (OpenAI). Supports multi-turn conversations using server-side context management.

Parameters:
- `message` (required): The message to send
- `model` (optional): Model to use (default: "gpt-5.1-2025-11-13")
- `system_prompt` (optional): System prompt for context (persists throughout conversation unless overridden)
- `temperature` (optional): 0.0 to 2.0 (uses model default if not specified)
- `max_tokens` (optional): Maximum response length
- `conversation_id` (optional): Conversation ID from a previous call to continue the conversation

**Multi-turn conversations:**
The tool returns a Conversation ID in the output (format: `chatgpt_...`). Pass this ID to `conversation_id` in subsequent calls to continue the conversation. The server stores conversation history in memory with automatic limits to prevent memory issues.

Conversation IDs are provider-specific and cannot be used across different providers (e.g., a ChatGPT conversation_id cannot be used with Claude).

**System prompt persistence:**
When you provide a `system_prompt` in the first message of a conversation, it will be automatically reused for all subsequent messages in that conversation unless explicitly overridden.

Example workflow:
1. First call: `query_chatgpt("Hello!")` → Returns `[Conversation ID: chatgpt_abc123def456]`
2. Follow-up: `query_chatgpt("How are you?", conversation_id="chatgpt_abc123def456")`

### query_claude
Send a message to Claude (Anthropic). Supports multi-turn conversations using server-side context management.

Parameters:
- `message` (required): The message to send
- `model` (optional): Model to use (default: "claude-sonnet-4-5-20250929")
- `system_prompt` (optional): System prompt for context (persists throughout conversation unless overridden)
- `temperature` (optional): 0.0 to 1.0 (uses model default if not specified)
- `max_tokens` (optional): Maximum response length (default: 4096)
- `conversation_id` (optional): Conversation ID from a previous call to continue the conversation

**Multi-turn conversations:**
The tool returns a Conversation ID in the output (format: `claude_...`). Pass this ID to `conversation_id` in subsequent calls to continue the conversation. The server stores conversation history in memory with automatic limits to prevent memory issues.

Conversation IDs are provider-specific and cannot be used across different providers (e.g., a Claude conversation_id cannot be used with Gemini).

**System prompt persistence:**
When you provide a `system_prompt` in the first message of a conversation, it will be automatically reused for all subsequent messages in that conversation unless explicitly overridden.

Example workflow:
1. First call: `query_claude("Hello!")` → Returns `[Conversation ID: claude_abc123def456]`
2. Follow-up: `query_claude("How are you?", conversation_id="claude_abc123def456")`

### query_gemini
Send a message to Gemini (Google). Supports multi-turn conversations using server-side context management.

Parameters:
- `message` (required): The message to send
- `model` (optional): Model to use (default: "gemini-2.5-flash")
- `system_prompt` (optional): System prompt for context (persists throughout conversation unless overridden)
- `temperature` (optional): 0.0 to 2.0 (uses model default if not specified)
- `max_tokens` (optional): Maximum response length (uses model default if not specified)
- `conversation_id` (optional): Conversation ID from a previous call to continue the conversation

**Multi-turn conversations:**
The tool returns a Conversation ID in the output (format: `gemini_...`). Pass this ID to `conversation_id` in subsequent calls to continue the conversation. The server stores conversation history in memory with automatic limits to prevent memory issues.

Conversation IDs are provider-specific and cannot be used across different providers (e.g., a Gemini conversation_id cannot be used with Claude).

**System prompt persistence:**
When you provide a `system_prompt` in the first message of a conversation, it will be automatically reused for all subsequent messages in that conversation unless explicitly overridden.

Example workflow:
1. First call: `query_gemini("Hello!")` → Returns `[Conversation ID: gemini_abc123def456]`
2. Follow-up: `query_gemini("How are you?", conversation_id="gemini_abc123def456")`

### export_conversation
Export a conversation to a markdown file. The conversation will be saved to the directory specified by the `LLM_CONVERSATIONS_DIR` environment variable.

Parameters:
- `conversation_id` (required): The conversation ID to export (e.g., "chatgpt_abc123def456")
- `calling_llm_name` (required): The name of the LLM making the calls (e.g., "Claude", "ChatGPT", "Gemini")

**Output format:**
The exported file will be a markdown file with the conversation ID as the filename (e.g., `chatgpt_abc123def456.md`). The conversation is formatted with alternating messages labeled by the participant names.
