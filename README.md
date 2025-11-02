# LLM MCP Server

A simple MCP (Model Context Protocol) server that provides tools to query Claude and ChatGPT.

## Features

- `query_claude`: Send messages to Claude (Anthropic API)
- `query_chatgpt`: Send messages to ChatGPT (OpenAI API)

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
Send a message to ChatGPT (OpenAI).

Parameters:
- `message` (required): The message to send
- `model` (optional): Model to use (default: "gpt-4.1-2025-04-14")
- `system_prompt` (optional): System prompt for context
- `temperature` (optional): 0.0 to 2.0 (default: 0.7)
- `max_tokens` (optional): Maximum response length

### query_claude
Send a message to Claude (Anthropic).

Parameters:
- `message` (required): The message to send
- `model` (optional): Model to use (default: "claude-sonnet-4-5-20250929")
- `system_prompt` (optional): System prompt for context
- `temperature` (optional): 0.0 to 1.0 (default: 1.0)
- `max_tokens` (optional): Maximum response length (default: 4096)
