#!/usr/bin/env python3
"""
Quick test to verify SupabaseChatHistory compatibility with LLM.py
"""

import os
import sys
import uuid

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_chat_history_compatibility():
    """Test that SupabaseChatHistory has the same interface as ChatHistory"""
    print("Testing chat history compatibility...")
    
    try:
        from supabase_chat_history import SupabaseChatHistory
        
        test_user_id = str(uuid.uuid4())
        
        # Create chat history instance
        chat = SupabaseChatHistory(test_user_id)
        
        # Test that it has the messages property
        if hasattr(chat, 'messages'):
            print("✓ SupabaseChatHistory has 'messages' property")
        else:
            print("✗ SupabaseChatHistory missing 'messages' property")
            return False
        
        # Test that messages property works like a list
        try:
            messages = chat.messages
            print(f"✓ messages property returns: {type(messages)} with {len(messages)} items")
        except Exception as e:
            print(f"✗ Error accessing messages property: {e}")
            return False
        
        # Test length operation (used by LLM.py)
        try:
            length = len(chat.messages)
            print(f"✓ len(chat.messages) works: {length}")
        except Exception as e:
            print(f"✗ Error with len(chat.messages): {e}")
            return False
        
        # Test adding a message and accessing via messages property
        try:
            chat.add_message("user", "test message")
            updated_messages = chat.messages
            print(f"✓ After adding message, messages has {len(updated_messages)} items")
        except Exception as e:
            print(f"✗ Error adding message or accessing updated messages: {e}")
            return False
        
        # Clean up
        chat.clear_history()
        print("✓ Test data cleaned up")
        
        return True
        
    except Exception as e:
        print(f"✗ Chat history compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_chat_history_compatibility()
    if success:
        print("\n🎉 Chat history compatibility test passed!")
    else:
        print("\n❌ Chat history compatibility test failed!")
    sys.exit(0 if success else 1)