#!/usr/bin/env python3
"""MCP Server for querying Claude and ChatGPT."""

import os
import asyncio
from typing import Optional
from dotenv import load_dotenv

from mcp.server import Server
from mcp.types import Tool, TextContent
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

# Load environment variables
load_dotenv()

# Initialize API clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Create MCP server
app = Server("llm-mcp-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="query_chatgpt",
            description="Send a message to ChatGPT and get a response",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to send to ChatGPT",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to use (default: gpt-4.1-2025-04-14)",
                        "default": "gpt-4.1-2025-04-14",
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "Optional system prompt for context",
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Sampling temperature 0.0-2.0 (default: 0.7)",
                        "default": 0.7,
                    },
                    "max_tokens": {
                        "type": "number",
                        "description": "Maximum tokens in response",
                    },
                },
                "required": ["message"],
            },
        ),
        Tool(
            name="query_claude",
            description="Send a message to Claude and get a response",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to send to Claude",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to use (default: claude-sonnet-4-5-20250929)",
                        "default": "claude-sonnet-4-5-20250929",
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "Optional system prompt for context",
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Sampling temperature 0.0-1.0 (default: 1.0)",
                        "default": 1.0,
                    },
                    "max_tokens": {
                        "type": "number",
                        "description": "Maximum tokens in response (default: 4096)",
                        "default": 4096,
                    },
                },
                "required": ["message"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    if name == "query_chatgpt":
        return await query_chatgpt(
            message=arguments["message"],
            model=arguments.get("model", "gpt-4.1-2025-04-14"),
            system_prompt=arguments.get("system_prompt"),
            temperature=arguments.get("temperature", 0.7),
            max_tokens=arguments.get("max_tokens"),
        )
    elif name == "query_claude":
        return await query_claude(
            message=arguments["message"],
            model=arguments.get("model", "claude-sonnet-4-5-20250929"),
            system_prompt=arguments.get("system_prompt"),
            temperature=arguments.get("temperature", 1.0),
            max_tokens=arguments.get("max_tokens", 4096),
        )
    else:
        raise ValueError(f"Unknown tool: {name}")


async def query_chatgpt(
    message: str,
    model: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> list[TextContent]:
    """Query ChatGPT."""
    # Build system prompt with AI-to-AI context
    base_context = "You are conversing with another AI assistant (Claude)."
    if system_prompt:
        final_system_prompt = f"{base_context}\n\n{system_prompt}"
    else:
        final_system_prompt = base_context

    messages = []
    messages.append({"role": "system", "content": final_system_prompt})
    messages.append({"role": "user", "content": message})

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens

    response = await openai_client.chat.completions.create(**kwargs)

    content = response.choices[0].message.content or ""
    usage_info = ""
    if response.usage:
        usage_info = f"\n\n[Usage: {response.usage.total_tokens} tokens ({response.usage.prompt_tokens} prompt, {response.usage.completion_tokens} completion)]"

    return [TextContent(type="text", text=f"{content}{usage_info}")]


async def query_claude(
    message: str,
    model: str,
    system_prompt: Optional[str] = None,
    temperature: float = 1.0,
    max_tokens: int = 4096,
) -> list[TextContent]:
    """Query Claude."""
    # Build system prompt with AI-to-AI context
    base_context = "You are conversing with another AI assistant (Claude)."
    if system_prompt:
        final_system_prompt = f"{base_context}\n\n{system_prompt}"
    else:
        final_system_prompt = base_context

    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": message}],
        "system": final_system_prompt,
    }

    response = await anthropic_client.messages.create(**kwargs)

    content = response.content[0].text
    usage_info = f"\n\n[Usage: {response.usage.input_tokens + response.usage.output_tokens} tokens ({response.usage.input_tokens} input, {response.usage.output_tokens} output)]"

    return [TextContent(type="text", text=f"{content}{usage_info}")]


async def main():
    """Run the MCP server."""
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
