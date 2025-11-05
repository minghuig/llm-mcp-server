"""In-memory conversation store with safeguards against OOM."""

import uuid
from collections import OrderedDict
from typing import Optional


class ConversationStore:
    """In-memory store for conversation contexts with size limits."""

    def __init__(self, max_conversations: int = 50):
        """Initialize the conversation store.

        Args:
            max_conversations: Maximum number of conversations to store (default: 50)
        """
        self.max_conversations = max_conversations
        self.conversations = OrderedDict()

    def create_context(self, context: list, provider: str, system_prompt: str) -> str:
        """Create a new conversation context and return its ID.

        Args:
            context: List of message objects representing the conversation
            provider: Provider name (e.g., "claude", "gemini", "chatgpt") to prefix the ID
            system_prompt: System prompt to store with the conversation

        Returns:
            context_id: Unique identifier for this conversation with provider prefix
        """
        context_id = f"ctx_{provider}_{uuid.uuid4().hex[:16]}"

        # Evict oldest conversation if at capacity
        if len(self.conversations) >= self.max_conversations:
            self.conversations.popitem(last=False)  # Remove oldest (FIFO)

        self.conversations[context_id] = {
            'system_prompt': system_prompt,
            'conversations': context
        }
        return context_id

    def get_context(self, context_id: str) -> Optional[dict]:
        """Retrieve conversation context by ID.

        Args:
            context_id: The conversation identifier

        Returns:
            Dict with 'conversations' and 'system_prompt' keys, or None if not found
        """
        # Move to end (mark as recently used for LRU)
        if context_id in self.conversations:
            self.conversations.move_to_end(context_id)
            return self.conversations[context_id]
        return None

    def update_context(self, context_id: str, new_messages: list, system_prompt: str):
        """Append new messages to an existing conversation.

        Args:
            context_id: The conversation identifier
            new_messages: List of new message objects to append
            system_prompt: System prompt to store
        """

        # Append new messages
        self.conversations[context_id]['conversations'].extend(new_messages)

        # Update system prompt
        self.conversations[context_id]['system_prompt'] = system_prompt

        # Move to end (mark as recently used)
        self.conversations.move_to_end(context_id)

    def clear(self):
        """Clear all stored conversations."""
        self.conversations.clear()

    def size(self) -> int:
        """Return the number of stored conversations."""
        return len(self.conversations)
