"""
Supabase-based chat history management for Hozie voice assistant.
Provides functionality to store, retrieve and manage conversation history in Supabase.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
import uuid
from supabase_client import supabase_client

class SupabaseChatHistory:
    """Manages chat history between the user and assistant using Supabase storage.
    
    This class provides methods to:
    - Store messages in the current conversation in Supabase
    - Retrieve the full conversation history from Supabase
    - Get recent context for the LLM
    - Clear conversation history
    """
    
    def __init__(self, user_id: str):
        """Initialize chat history manager.
        
        Args:
            user_id: The ID (UUID string) of the user whose chat history to manage
        """
        self.user_id = user_id
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add a message to the chat history.
        
        Args:
            role: The role of the message sender (e.g., "user", "assistant")
            content: The content of the message
            metadata: Optional additional data to store with the message
                     (e.g., audio file path, confidence)
                     
        Returns:
            True if message was saved successfully, False otherwise
        """
        try:
            message_data = {
                'id': str(uuid.uuid4()),
                'user_id': self.user_id,
                'message_type': role,
                'content': content,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'metadata': metadata or {}
            }
            
            result = supabase_client.client.table('chat_history').insert(message_data).execute()
            return len(result.data) > 0
        except Exception as e:
            print(f"Error adding message to chat history: {e}")
            return False
    
    def get_history(self, max_messages: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve chat history.
        
        Args:
            max_messages: Optional limit to the number of most recent messages to return
            
        Returns:
            List of message objects in the chat history
        """
        try:
            query = supabase_client.client.table('chat_history')\
                .select('*')\
                .eq('user_id', self.user_id)\
                .order('timestamp', desc=False)
            
            if max_messages:
                # Get the most recent messages
                query = query.limit(max_messages)
            
            result = query.execute()
            
            # Convert to the expected format
            messages = []
            for row in result.data:
                message = {
                    'role': row['message_type'],
                    'content': row['content'],
                    'timestamp': row['timestamp']
                }
                if row.get('metadata'):
                    message['metadata'] = row['metadata']
                messages.append(message)
            
            # If we limited messages, we want the most recent ones
            if max_messages and len(messages) > max_messages:
                messages = messages[-max_messages:]
                
            return messages
        except Exception as e:
            print(f"Error retrieving chat history: {e}")
            return []
    
    def get_formatted_history(self, max_messages: Optional[int] = None) -> str:
        """Get formatted chat history for review.
        
        Args:
            max_messages: Optional limit to the number of most recent messages to return
            
        Returns:
            Formatted string representation of the chat history
        """
        messages = self.get_history(max_messages)
        formatted = []
        
        for msg in messages:
            timestamp = msg.get("timestamp", "")
            if timestamp:
                # Try to parse and format timestamp
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                except (ValueError, TypeError):
                    time_str = timestamp
            else:
                time_str = ""
            
            formatted.append(f"[{time_str}] {msg['role'].title()}: {msg['content']}")
            
        return "\n".join(formatted)
    
    def get_context_for_llm(self, max_messages: int = 10) -> List[Dict[str, str]]:
        """Get recent conversation formatted for input to an LLM.
        
        Args:
            max_messages: Maximum number of recent messages to include
            
        Returns:
            List of message objects formatted for LLM context
        """
        recent_messages = self.get_history(max_messages)
        
        # Format for LLM context (simplified format with just role and content)
        llm_context = []
        for msg in recent_messages:
            llm_context.append({
                "role": msg["role"],
                "content": msg["content"]
            })
            
        return llm_context
    
    def clear_history(self) -> bool:
        """Clear all chat history for this user.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            result = supabase_client.client.table('chat_history')\
                .delete()\
                .eq('user_id', self.user_id)\
                .execute()
            return True
        except Exception as e:
            print(f"Error clearing chat history: {e}")
            return False
    
    def get_message_count(self) -> int:
        """Get the total number of messages for this user.
        
        Returns:
            Number of messages in history
        """
        try:
            result = supabase_client.client.table('chat_history')\
                .select('id', count='exact')\
                .eq('user_id', self.user_id)\
                .execute()
            return result.count if result.count is not None else 0
        except Exception as e:
            print(f"Error getting message count: {e}")
            return 0
    
    def get_messages_since(self, since_timestamp: str) -> List[Dict[str, Any]]:
        """Get all messages since a specific timestamp.
        
        Args:
            since_timestamp: ISO timestamp string
            
        Returns:
            List of messages since the timestamp
        """
        try:
            result = supabase_client.client.table('chat_history')\
                .select('*')\
                .eq('user_id', self.user_id)\
                .gte('timestamp', since_timestamp)\
                .order('timestamp', desc=False)\
                .execute()
            
            messages = []
            for row in result.data:
                message = {
                    'role': row['message_type'],
                    'content': row['content'],
                    'timestamp': row['timestamp']
                }
                if row.get('metadata'):
                    message['metadata'] = row['metadata']
                messages.append(message)
                
            return messages
        except Exception as e:
            print(f"Error retrieving messages since {since_timestamp}: {e}")
            return []
    
    def __len__(self) -> int:
        """Get the number of messages in history."""
        return self.get_message_count()
    
    @property
    def messages(self):
        """
        Compatibility property to mimic the file-based ChatHistory.messages attribute.
        Returns the full message history in the same format.
        """
        return self.get_history()

    @classmethod
    def migrate_from_file(cls, file_path: str, user_id: str) -> bool:
        """
        Migrate chat history from a JSON file to Supabase
        
        Args:
            file_path: Path to the JSON file containing chat history
            user_id: User ID to associate with the messages
            
        Returns:
            True if migration was successful, False otherwise
        """
        import json
        import os
        
        if not os.path.exists(file_path):
            return False
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
            
            messages = history_data.get('messages', [])
            chat_history = cls(user_id)
            
            success_count = 0
            for msg in messages:
                if chat_history.add_message(
                    role=msg.get('role', 'user'),
                    content=msg.get('content', ''),
                    metadata=msg.get('metadata')
                ):
                    success_count += 1
            
            print(f"Migrated {success_count}/{len(messages)} messages for user {user_id}")
            return success_count == len(messages)
            
        except Exception as e:
            print(f"Error migrating chat history from {file_path}: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Create a chat history instance for a user
    user_id = "test_user_123"
    chat = SupabaseChatHistory(user_id)
    
    # Add some messages
    chat.add_message("user", "Hello, how are you?")
    chat.add_message("assistant", "I'm doing well, thank you! How can I help you today?")
    
    # Display formatted history
    print(chat.get_formatted_history())
    
    # Get context for LLM
    context = chat.get_context_for_llm()
    print(f"LLM Context: {context}")