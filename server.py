#!/usr/bin/env python3
"""MCP Server for querying Claude, ChatGPT, and Gemini."""

import os
import asyncio
from typing import Optional
from dotenv import load_dotenv

from mcp.server import Server
from mcp.types import Tool, TextContent
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

# Initialize API clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Create MCP server
app = Server("llm-mcp-server")


def build_system_prompt(system_prompt: Optional[str] = None) -> str:
    """Build system prompt with AI-to-AI context."""
    base_context = "You are conversing with another AI assistant (Claude)."
    if system_prompt:
        return f"{base_context}\n\n{system_prompt}"
    return base_context


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="query_chatgpt",
            description="Send a message to ChatGPT and get a response. Returns a Response ID that can be used to continue the conversation in follow-up calls.",
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
                    "previous_response_id": {
                        "type": "string",
                        "description": "Optional ID from a previous response to continue that conversation. Use the Response ID returned in a previous call to maintain conversation history.",
                    },
                },
                "required": ["message"],
            },
        ),
        Tool(
            name="query_claude",
            description="Send a message to Claude and get a response. Returns conversation context that can be used to continue the conversation in follow-up calls.",
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
                    "context": {
                        "type": "array",
                        "description": "Optional conversation history (array of message objects with 'role' and 'content'). Use the context returned in a previous call to maintain conversation history.",
                        "items": {
                            "type": "object"
                        }
                    },
                },
                "required": ["message"],
            },
        ),
        Tool(
            name="query_gemini",
            description="Send a message to Gemini and get a response. Returns conversation context that can be used to continue the conversation in follow-up calls.",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to send to Gemini",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to use (default: gemini-2.5-flash)",
                        "default": "gemini-2.5-flash",
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "Optional system prompt for context",
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Sampling temperature 0.0-2.0 (uses model default if not specified)",
                    },
                    "max_tokens": {
                        "type": "number",
                        "description": "Maximum tokens in response (uses model default if not specified)",
                    },
                    "context": {
                        "type": "array",
                        "description": "Optional conversation history (array of content objects with 'role' and 'parts'). Use the context returned in a previous call to maintain conversation history.",
                        "items": {
                            "type": "object"
                        }
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
            previous_response_id=arguments.get("previous_response_id"),
        )
    elif name == "query_claude":
        return await query_claude(
            message=arguments["message"],
            model=arguments.get("model", "claude-sonnet-4-5-20250929"),
            system_prompt=arguments.get("system_prompt"),
            temperature=arguments.get("temperature", 1.0),
            max_tokens=arguments.get("max_tokens", 4096),
            context=arguments.get("context"),
        )
    elif name == "query_gemini":
        return await query_gemini(
            message=arguments["message"],
            model=arguments.get("model", "gemini-2.5-flash"),
            system_prompt=arguments.get("system_prompt"),
            temperature=arguments.get("temperature"),
            max_tokens=arguments.get("max_tokens"),
            context=arguments.get("context"),
        )
    else:
        raise ValueError(f"Unknown tool: {name}")


async def query_chatgpt(
    message: str,
    model: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    previous_response_id: Optional[str] = None,
) -> list[TextContent]:
    """Query ChatGPT using Responses API for stateful conversations."""

    # Build system prompt with AI-to-AI context
    instructions = build_system_prompt(system_prompt)

    # Build kwargs for responses.create()
    kwargs = {
        "model": model,
        "instructions": instructions,
        "input": message,
        "temperature": temperature,
        "store": True,  # Store conversation for continuation
    }

    if max_tokens:
        kwargs["max_tokens"] = max_tokens

    # If continuing a conversation, pass previous_response_id
    if previous_response_id:
        kwargs["previous_response_id"] = previous_response_id

    # Call the Responses API
    response = await openai_client.responses.create(**kwargs)

    # Get the text content to display
    content = response.output_text

    # Format usage info
    usage_info = f"\n\n[Usage: {response.usage.total_tokens} tokens ({response.usage.input_tokens} input, {response.usage.output_tokens} output)]"

    # Return content with response ID for continuation
    return [TextContent(type="text", text=f"{content}{usage_info}\n\n[Response ID: {response.id}]")]


async def query_claude(
    message: str,
    model: str,
    system_prompt: Optional[str] = None,
    temperature: float = 1.0,
    max_tokens: int = 4096,
    context: Optional[list] = None,
) -> list[TextContent]:
    """Query Claude using stateful conversations with message history."""
    import json

    # Build system prompt with AI-to-AI context
    system_prompt = build_system_prompt(system_prompt)

    # Initialize or use existing context
    if context is None:
        context = []

    # Add the new user message to context
    context.append({"role": "user", "content": message})

    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": context,
        "system": system_prompt,
    }

    response = await anthropic_client.messages.create(**kwargs)

    content = response.content[0].text

    # Add assistant's response to context
    context.append({"role": "assistant", "content": content})

    usage_info = f"\n\n[Usage: {response.usage.input_tokens + response.usage.output_tokens} tokens ({response.usage.input_tokens} input, {response.usage.output_tokens} output)]"

    # Return both content and updated context
    context_json = json.dumps(context)
    return [TextContent(type="text", text=f"{content}{usage_info}\n\n[Context: {context_json}]")]


async def query_gemini(
    message: str,
    model: str,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    context: Optional[list] = None,
) -> list[TextContent]:
    """Query Gemini using stateful conversations with message history."""
    import json

    # Build system prompt with AI-to-AI context
    system_prompt = build_system_prompt(system_prompt)

    # Initialize or use existing context
    if context is None:
        context = []

    # Add the new user message to context
    # Gemini Content format: {"role": "user", "parts": [{"text": "..."}]}
    context.append({"role": "user", "parts": [{"text": message}]})

    # Build GenerateContentConfig - only include params if explicitly provided
    config_params = {"system_instruction": system_prompt}
    if temperature is not None:
        config_params["temperature"] = temperature
    if max_tokens is not None:
        config_params["max_output_tokens"] = max_tokens

    config = types.GenerateContentConfig(**config_params)

    # Call Gemini API
    response = google_client.models.generate_content(
        model=model,
        contents=context,
        config=config,
    )

    # Extract text from response
    content = response.text

    # Add model's response to context
    # Gemini uses "model" role for assistant responses
    context.append({"role": "model", "parts": [{"text": content}]})

    # Format usage info
    usage_info = ""
    if response.usage_metadata:
        total = response.usage_metadata.total_token_count
        prompt = response.usage_metadata.prompt_token_count
        completion = response.usage_metadata.candidates_token_count
        usage_info = f"\n\n[Usage: {total} tokens ({prompt} input, {completion} output)]"

    # Return both content and updated context
    context_json = json.dumps(context)
    return [TextContent(type="text", text=f"{content}{usage_info}\n\n[Context: {context_json}]")]


async def main():
    """Run the MCP server."""
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
