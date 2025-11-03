"""In-memory conversation store with safeguards against OOM."""

import uuid
from collections import OrderedDict
from typing import Optional


class ConversationStore:
    """In-memory store for conversation contexts with size limits."""

    def __init__(
        self, max_conversations: int = 100, max_turns_per_conversation: int = 50
    ):
        """Initialize the conversation store.

        Args:
            max_conversations: Maximum number of conversations to store
            max_turns_per_conversation: Maximum number of turns per conversation
        """
        self.max_conversations = max_conversations
        self.max_turns_per_conversation = max_turns_per_conversation
        self.conversations = OrderedDict()

    def save_context(self, context: list, provider: str) -> str:
        """Save a new conversation context and return its ID.

        Args:
            context: List of message objects representing the conversation
            provider: Provider name (e.g., "claude", "gemini") to prefix the ID

        Returns:
            context_id: Unique identifier for this conversation with provider prefix
        """
        context_id = f"ctx_{provider}_{uuid.uuid4().hex[:16]}"

        # Enforce per-conversation turn limit
        if len(context) > self.max_turns_per_conversation:
            context = context[-self.max_turns_per_conversation :]

        # Evict oldest conversation if at capacity
        if len(self.conversations) >= self.max_conversations:
            self.conversations.popitem(last=False)  # Remove oldest (FIFO)

        self.conversations[context_id] = context
        return context_id

    def get_context(self, context_id: str) -> Optional[list]:
        """Retrieve conversation context by ID.

        Args:
            context_id: The conversation identifier

        Returns:
            List of message objects, or None if not found
        """
        # Move to end (mark as recently used for LRU)
        if context_id in self.conversations:
            self.conversations.move_to_end(context_id)
            return self.conversations[context_id]
        return None

    def update_context(self, context_id: str, new_messages: list) -> bool:
        """Append new messages to an existing conversation.

        Args:
            context_id: The conversation identifier
            new_messages: List of new message objects to append

        Returns:
            True if successful, False if context_id not found
        """
        if context_id not in self.conversations:
            return False

        # Append new messages
        self.conversations[context_id].extend(new_messages)

        # Enforce turn limit by truncating old messages
        if len(self.conversations[context_id]) > self.max_turns_per_conversation:
            self.conversations[context_id] = self.conversations[context_id][
                -self.max_turns_per_conversation :
            ]

        # Move to end (mark as recently used)
        self.conversations.move_to_end(context_id)
        return True

    def clear(self):
        """Clear all stored conversations."""
        self.conversations.clear()

    def size(self) -> int:
        """Return the number of stored conversations."""
        return len(self.conversations)
