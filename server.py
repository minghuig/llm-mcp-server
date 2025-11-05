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

from conversation_store import ConversationStore

# Load environment variables
load_dotenv()

# Initialize API clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize conversation store
conversation_store = ConversationStore(max_conversations=50)

# Create MCP server
app = Server("llm-mcp-server")


def build_system_prompt(system_prompt: Optional[str] = None) -> str:
    """Build system prompt with AI-to-AI context."""
    base_context = "You are conversing with another AI assistant through an MCP server."
    if system_prompt:
        return f"{base_context}\n\n{system_prompt}"
    return base_context


def get_validated_conversation(
    conversation_id: Optional[str], provider: str
) -> tuple[Optional[dict], Optional[str]]:
    """Retrieve and validate conversation.

    Args:
        conversation_id: Optional conversation ID to retrieve
        provider: Expected provider name (e.g., "claude", "gemini", "chatgpt")

    Returns:
        Tuple of (conversation, error_message)
        - conversation: Dict with 'messages' and 'system_prompt' keys if exists, None otherwise
        - error_message: Error string if validation failed, None otherwise
    """
    if not conversation_id:
        return None, None

    # Validate provider prefix
    expected_prefix = f"{provider}_"
    if not conversation_id.startswith(expected_prefix):
        error_msg = (
            f"Error: Invalid conversation_id for {provider.title()}. "
            f"Expected conversation_id starting with '{expected_prefix}', got '{conversation_id}'. "
            f"This conversation_id appears to be for a different provider."
        )
        return None, error_msg

    # Try to retrieve conversation    
    return conversation_store.get_conversation(conversation_id), None


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="query_chatgpt",
            description="Send a message to ChatGPT and get a response. Returns a Conversation ID that can be used to continue the conversation in follow-up calls.",
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
                        "description": "Optional system prompt for context. Persists throughout the conversation unless explicitly overridden.",
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Sampling temperature 0.0-2.0 (uses model default if not specified)",
                    },
                    "max_tokens": {
                        "type": "number",
                        "description": "Maximum tokens in response",
                    },
                    "conversation_id": {
                        "type": "string",
                        "description": "Optional ID from a previous response to continue that conversation. Use the Conversation ID returned in a previous call (found in the output after '[Conversation ID: ...]') to maintain conversation history.",
                    },
                },
                "required": ["message"],
            },
        ),
        Tool(
            name="query_claude",
            description="Send a message to Claude and get a response. Returns a Conversation ID that can be used to continue the conversation in follow-up calls.",
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
                        "description": "Optional system prompt for context. Persists throughout the conversation unless explicitly overridden.",
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Sampling temperature 0.0-1.0 (uses model default if not specified)",
                    },
                    "max_tokens": {
                        "type": "number",
                        "description": "Maximum tokens in response (default: 4096)",
                        "default": 4096,
                    },
                    "conversation_id": {
                        "type": "string",
                        "description": "Optional ID from a previous response to continue that conversation. Use the Conversation ID returned in a previous call (found in the output after '[Conversation ID: ...]') to maintain conversation history.",
                    },
                },
                "required": ["message"],
            },
        ),
        Tool(
            name="query_gemini",
            description="Send a message to Gemini and get a response. Returns a Conversation ID that can be used to continue the conversation in follow-up calls.",
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
                        "description": "Optional system prompt for context. Persists throughout the conversation unless explicitly overridden.",
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Sampling temperature 0.0-2.0 (uses model default if not specified)",
                    },
                    "max_tokens": {
                        "type": "number",
                        "description": "Maximum tokens in response (uses model default if not specified)",
                    },
                    "conversation_id": {
                        "type": "string",
                        "description": "Optional ID from a previous response to continue that conversation. Use the Conversation ID returned in a previous call (found in the output after '[Conversation ID: ...]') to maintain conversation history.",
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
            temperature=arguments.get("temperature"),
            max_tokens=arguments.get("max_tokens"),
            conversation_id=arguments.get("conversation_id"),
        )
    elif name == "query_claude":
        return await query_claude(
            message=arguments["message"],
            model=arguments.get("model", "claude-sonnet-4-5-20250929"),
            system_prompt=arguments.get("system_prompt"),
            temperature=arguments.get("temperature"),
            max_tokens=arguments.get("max_tokens", 4096),
            conversation_id=arguments.get("conversation_id"),
        )
    elif name == "query_gemini":
        return await query_gemini(
            message=arguments["message"],
            model=arguments.get("model", "gemini-2.5-flash"),
            system_prompt=arguments.get("system_prompt"),
            temperature=arguments.get("temperature"),
            max_tokens=arguments.get("max_tokens"),
            conversation_id=arguments.get("conversation_id"),
        )
    else:
        raise ValueError(f"Unknown tool: {name}")


async def query_chatgpt(
    message: str,
    model: str,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    conversation_id: Optional[str] = None,
) -> list[TextContent]:
    """Query ChatGPT using Responses API for stateful conversations."""

    # Retrieve and validate existing conversation
    conversation, error_msg = get_validated_conversation(conversation_id, "chatgpt")
    if error_msg:
        return [TextContent(type="text", text=error_msg)]
    
    if conversation and not system_prompt:
        system_prompt = conversation['system_prompt']
    else:
        system_prompt = build_system_prompt(system_prompt)

    messages = (conversation['messages'] if conversation else []) + [{"role": "user", "content": message}]

    # Build kwargs for responses.create()
    kwargs = {
        "model": model,
        "instructions": system_prompt,
        "input": messages,
    }

    if temperature is not None:
        kwargs["temperature"] = temperature

    if max_tokens:
        kwargs["max_output_tokens"] = max_tokens

    # Call the Responses API
    response = await openai_client.responses.create(**kwargs)

    # Get the text content to display
    content = response.output_text

    # Prepare new messages to add to conversation
    new_messages = [
        {"role": "user", "content": message},
        {"role": "assistant", "content": content}
    ]

    # Save or update conversation in store
    if conversation is None:
        conversation_id = conversation_store.create_conversation(new_messages, "chatgpt", system_prompt)
    else:
        conversation_store.update_conversation(conversation_id, new_messages, system_prompt)

    # Format usage info
    usage_info = f"\n\n[Usage: {response.usage.total_tokens} tokens ({response.usage.input_tokens} input, {response.usage.output_tokens} output)]"

    # Return content with conversation ID for continuation
    return [TextContent(type="text", text=f"{content}{usage_info}\n\n[Conversation ID: {conversation_id}]")]


async def query_claude(
    message: str,
    model: str,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: int = 4096,
    conversation_id: Optional[str] = None,
) -> list[TextContent]:
    """Query Claude using stateful conversations with conversation IDs."""

    # Retrieve and validate existing conversation
    conversation, error_msg = get_validated_conversation(conversation_id, "claude")
    if error_msg:
        return [TextContent(type="text", text=error_msg)]
    
    if conversation and not system_prompt:
        system_prompt = conversation['system_prompt']
    else:
        system_prompt = build_system_prompt(system_prompt)

    # Build messages for API call
    messages = (conversation['messages'] if conversation else []) + [{"role": "user", "content": message}]

    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        "system": system_prompt,
    }

    if temperature is not None:
        kwargs["temperature"] = temperature

    response = await anthropic_client.messages.create(**kwargs)
    content = response.content[0].text

    # Prepare new messages to add to conversation
    new_messages = [
        {"role": "user", "content": message},
        {"role": "assistant", "content": content}
    ]

    # Save or update conversation in store
    if conversation is None:
        conversation_id = conversation_store.create_conversation(new_messages, "claude", system_prompt)
    else:
        conversation_store.update_conversation(conversation_id, new_messages, system_prompt)

    usage_info = f"\n\n[Usage: {response.usage.input_tokens + response.usage.output_tokens} tokens ({response.usage.input_tokens} input, {response.usage.output_tokens} output)]"

    # Return content with conversation ID
    return [TextContent(type="text", text=f"{content}{usage_info}\n\n[Conversation ID: {conversation_id}]")]


async def query_gemini(
    message: str,
    model: str,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    conversation_id: Optional[str] = None,
) -> list[TextContent]:
    """Query Gemini using stateful conversations with conversation IDs."""
    # Retrieve and validate existing conversation
    conversation, error_msg = get_validated_conversation(conversation_id, "gemini")
    if error_msg:
        return [TextContent(type="text", text=error_msg)]
    
    if conversation and not system_prompt:
        system_prompt = conversation['system_prompt']
    else:
        system_prompt = build_system_prompt(system_prompt)

    # Build contents for API call
    # Gemini Content format: {"role": "user", "parts": [{"text": "..."}]}
    contents = (conversation['messages'] if conversation else []) + [{"role": "user", "parts": [{"text": message}]}]

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
        contents=contents,
        config=config,
    )

    # Extract text from response
    content = response.text

    # Prepare new messages to add to conversation
    # Gemini uses "model" role for assistant responses
    new_messages = [
        {"role": "user", "parts": [{"text": message}]},
        {"role": "model", "parts": [{"text": content}]}
    ]

    # Save or update conversation in store
    if conversation is None:
        conversation_id = conversation_store.create_conversation(new_messages, "gemini", system_prompt)
    else:
        conversation_store.update_conversation(conversation_id, new_messages, system_prompt)

    # Format usage info
    usage_info = ""
    if response.usage_metadata:
        total = response.usage_metadata.total_token_count
        prompt = response.usage_metadata.prompt_token_count
        completion = response.usage_metadata.candidates_token_count
        usage_info = f"\n\n[Usage: {total} tokens ({prompt} input, {completion} output)]"

    # Return content with conversation ID
    return [TextContent(type="text", text=f"{content}{usage_info}\n\n[Conversation ID: {conversation_id}]")]


async def main():
    """Run the MCP server."""
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
