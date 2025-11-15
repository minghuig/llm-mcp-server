#!/usr/bin/env python3
"""MCP Server for querying Claude, ChatGPT, and Gemini."""

import os
import asyncio
from typing import Optional
from dotenv import load_dotenv

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from google import genai
from google.genai import types

from conversation_store import ConversationStore

# Load environment variables
load_dotenv()

# Initialize API clients (only if API keys are present)
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY")) if os.getenv("ANTHROPIC_API_KEY") else None
google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY")) if os.getenv("GOOGLE_API_KEY") else None

# Initialize conversation store
conversation_store = ConversationStore(max_conversations=50)

# Get conversations export directory (expand ~ if present)
conversations_dir = os.path.expanduser(os.getenv("LLM_CONVERSATIONS_DIR", "~/Documents/llm_conversations"))

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
    """Retrieve and validate conversation."""

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
                        "description": "Model to use (default: gpt-5.1-2025-11-13)",
                        "default": "gpt-5.1-2025-11-13",
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
        Tool(
            name="export_conversation",
            description="Export a conversation to a markdown file. The conversation will be saved to the directory specified by LLM_CONVERSATIONS_DIR environment variable.",
            inputSchema={
                "type": "object",
                "properties": {
                    "conversation_id": {
                        "type": "string",
                        "description": "The conversation ID to export (e.g., 'chatgpt_abc123def456')",
                    },
                    "calling_llm_name": {
                        "type": "string",
                        "description": "The name of the LLM making the calls (e.g., 'Claude', 'ChatGPT', 'Gemini')",
                    },
                },
                "required": ["conversation_id", "calling_llm_name"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""

    if name == "query_chatgpt":
        return await query_chatgpt(
            message=arguments["message"],
            model=arguments.get("model", "gpt-5.1-2025-11-13"),
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
    elif name == "export_conversation":
        return export_conversation(
            conversation_id=arguments["conversation_id"],
            calling_llm_name=arguments["calling_llm_name"],
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

    # Check if ChatGPT is configured
    if openai_client is None:
        return [TextContent(type="text", text="Error: ChatGPT is not configured. Please add OPENAI_API_KEY to your .env file.")]

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

    # Check if Claude is configured
    if anthropic_client is None:
        return [TextContent(type="text", text="Error: Claude is not configured. Please add ANTHROPIC_API_KEY to your .env file.")]

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

    # Check if Gemini is configured
    if google_client is None:
        return [TextContent(type="text", text="Error: Gemini is not configured. Please add GOOGLE_API_KEY to your .env file.")]

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


def export_conversation(conversation_id: str, calling_llm_name: str) -> list[TextContent]:
    """Export a conversation to a markdown file."""

    # Get the conversation
    conversation = conversation_store.get_conversation(conversation_id)
    if conversation is None:
        return [TextContent(type="text", text=f"Error: Conversation '{conversation_id}' not found.")]

    # Extract provider from conversation_id (e.g., "chatgpt" from "chatgpt_abc123")
    provider = conversation_id.split("_")[0]

    # Map provider names to display names
    provider_display_names = {
        "chatgpt": "ChatGPT",
        "claude": "Claude",
        "gemini": "Gemini"
    }
    responding_llm_name = provider_display_names.get(provider, provider.title())

    # Build markdown content
    lines = [f"# Conversation: {conversation_id}\n"]

    messages = conversation['messages']
    for msg in messages:
        # Extract content based on message format
        if 'role' in msg:
            role = msg['role']
            if 'content' in msg:
                # ChatGPT/Claude format
                content = msg['content']
            elif 'parts' in msg:
                # Gemini format
                content = msg['parts'][0]['text']
            else:
                content = str(msg)

            # Determine which LLM sent this message
            if role == "user":
                llm_name = calling_llm_name
            elif role in ["assistant", "model"]:
                llm_name = responding_llm_name
            else:
                llm_name = role.title()

            lines.append(f"**{llm_name}:** {content}\n")

    markdown_content = "\n".join(lines)

    # Create directory if it doesn't exist
    os.makedirs(conversations_dir, exist_ok=True)

    # Write to file
    file_path = os.path.join(conversations_dir, f"{conversation_id}.md")
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        return [TextContent(type="text", text=f"Conversation exported successfully to: {file_path}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error writing file: {str(e)}")]


async def main():
    """Run the MCP server."""
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
