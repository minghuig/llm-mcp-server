"""In-memory conversation store with safeguards against OOM."""

import uuid
from collections import OrderedDict
from typing import Optional


class ConversationStore:
    """In-memory store for conversations with size limits."""

    def __init__(self, max_conversations: int = 50):
        """Initialize the conversation store.

        Args:
            max_conversations: Maximum number of conversations to store (default: 50)
        """
        self.max_conversations = max_conversations
        self.conversations = OrderedDict()

    def create_conversation(self, messages: list, provider: str, system_prompt: str) -> str:
        """Create a new conversation and return its ID.

        Args:
            messages: List of message objects representing the conversation
            provider: Provider name (e.g., "claude", "gemini", "chatgpt") to prefix the ID
            system_prompt: System prompt to store with the conversation

        Returns:
            conversation_id: Unique identifier for this conversation with provider prefix
        """
        conversation_id = f"{provider}_{uuid.uuid4().hex[:16]}"

        # Evict oldest conversation if at capacity
        if len(self.conversations) >= self.max_conversations:
            self.conversations.popitem(last=False)  # Remove oldest (FIFO)

        self.conversations[conversation_id] = {
            'system_prompt': system_prompt,
            'messages': messages
        }
        return conversation_id

    def get_conversation(self, conversation_id: str) -> Optional[dict]:
        """Retrieve conversation by ID.

        Args:
            conversation_id: The conversation identifier

        Returns:
            Dict with 'messages' and 'system_prompt' keys, or None if not found
        """
        # Move to end (mark as recently used for LRU)
        if conversation_id in self.conversations:
            self.conversations.move_to_end(conversation_id)
            return self.conversations[conversation_id]
        return None

    def update_conversation(self, conversation_id: str, new_messages: list, system_prompt: str):
        """Append new messages to an existing conversation.

        Args:
            conversation_id: The conversation identifier
            new_messages: List of new message objects to append
            system_prompt: System prompt to store
        """

        # Append new messages
        self.conversations[conversation_id]['messages'].extend(new_messages)

        # Update system prompt
        self.conversations[conversation_id]['system_prompt'] = system_prompt

        # Move to end (mark as recently used)
        self.conversations.move_to_end(conversation_id)

    def clear(self):
        """Clear all stored conversations."""
        self.conversations.clear()

    def size(self) -> int:
        """Return the number of stored conversations."""
        return len(self.conversations)
