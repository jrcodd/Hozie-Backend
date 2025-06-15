from __future__ import annotations
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone


class ChatHistory:
    """
    ChatHistory manages the conversation history between a user and hozie so it can answer follow up questions.
    """
    
    def __init__(self):
        """
        Initialize chat history
        """
        self.messages = []
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a message to the chat history.
        
        Args:
            role: The role of the message sender (e.g., "user", "bot")
            content: The content of the message
            metadata: Optional additional data to store with the message
                     (e.g., timestamp, audio file path, confidence)
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if metadata:
            message["metadata"] = metadata
            
        self.messages.append(message)
    
    def get_history(self, max_messages: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve chat history.
        
        Args:
            max_messages: Optional limit to the number of most recent messages to return (should be around 3 usually my follow ups dont go that long)
            
        Returns:
            List of message objects in the chat history
        """
        if max_messages:
            return self.messages[-max_messages:]
        return self.messages.copy()

    def clear_history(self) -> None:
        """
        Clear all chat history.
        """
        self.messages = []
         
    def __len__(self) -> int:
        """
        Get the number of messages in history.

        Returns:
            int: The number of messages in history.
        """
        return len(self.messages)
