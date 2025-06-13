#!/usr/bin/env python3
"""
Test script to verify Supabase integration is working correctly.
Run this after setting up your Supabase environment variables.
"""

import os
import sys
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_supabase_client():
    """Test basic Supabase client connection"""
    print("Testing Supabase client connection...")
    
    try:
        from supabase_client import supabase_client
        
        # Test basic connection
        response = supabase_client.client.table('memory_nodes').select('*').limit(1).execute()
        print("✓ Supabase client connection successful")
        return True
    except Exception as e:
        print(f"✗ Supabase client connection failed: {e}")
        return False

def test_database_setup():
    """Test database table creation"""
    print("Testing database setup...")
    
    try:
        from database_setup import create_tables
        
        result = create_tables()
        if result:
            print("✓ Database tables created/verified successfully")
            return True
        else:
            print("✗ Database setup failed")
            return False
    except Exception as e:
        print(f"✗ Database setup error: {e}")
        return False

def test_topic_node():
    """Test SupabaseTopicNode functionality"""
    print("Testing SupabaseTopicNode...")
    
    try:
        from supabase_topic_node import SupabaseTopicNode
        
        # No longer need user_id for global brain
        
        # Get global root node
        root = SupabaseTopicNode.get_global_root()
        print(f"✓ Global root node created/retrieved: {root.topic}")
        
        # Add a child node
        child = root.add_child("Test Topic", data={"test": "data"})
        print(f"✓ Child node created: {child.topic}")
        
        # Find the child
        found_child = root.find_child("Test Topic")
        if found_child:
            print("✓ Child node found successfully")
        else:
            print("✗ Child node not found")
            return False
        
        # Update data
        child.update_data({"updated": "data", "timestamp": datetime.now().isoformat()})
        print("✓ Child node data updated")
        
        # Test tree traversal
        path = child.get_path()
        print(f"✓ Node path: {' -> '.join(path)}")
        
        # Clean up test data
        child.delete()
        print("✓ Test data cleaned up")
        
        return True
        
    except Exception as e:
        print(f"✗ SupabaseTopicNode test failed: {e}")
        return False

def test_chat_history():
    """Test SupabaseChatHistory functionality"""
    print("Testing SupabaseChatHistory...")
    
    try:
        from supabase_chat_history import SupabaseChatHistory
        
        test_user_id = "test_user_123"
        
        # Create chat history instance
        chat = SupabaseChatHistory(test_user_id)
        
        # Add test messages
        success1 = chat.add_message("user", "Hello, this is a test message")
        success2 = chat.add_message("assistant", "This is a test response")
        
        if success1 and success2:
            print("✓ Messages added successfully")
        else:
            print("✗ Failed to add messages")
            return False
        
        # Retrieve messages
        messages = chat.get_history(max_messages=5)
        if len(messages) >= 2:
            print(f"✓ Retrieved {len(messages)} messages")
        else:
            print("✗ Failed to retrieve messages")
            return False
        
        # Test formatted history
        formatted = chat.get_formatted_history()
        if formatted:
            print("✓ Formatted history generated")
        else:
            print("✗ Failed to generate formatted history")
        
        # Test LLM context
        context = chat.get_context_for_llm()
        if context and len(context) >= 2:
            print(f"✓ LLM context generated with {len(context)} messages")
        else:
            print("✗ Failed to generate LLM context")
            return False
        
        # Clean up test data
        chat.clear_history()
        print("✓ Test chat history cleaned up")
        
        return True
        
    except Exception as e:
        print(f"✗ SupabaseChatHistory test failed: {e}")
        return False

def test_brain_integration():
    """Test Supabase-integrated Brain"""
    print("Testing Supabase-integrated Brain...")
    
    try:
        from LLM import Brain
        
        # Check if API key is available
        api_key = os.environ.get('MISTRAL_API_KEY')
        if not api_key:
            print("⚠ MISTRAL_API_KEY not found, skipping Brain test")
            return True
        
        # Create brain instance with Supabase storage
        brain = Brain(use_supabase=True)
        print("✓ Global Supabase-integrated Brain instance created")
        
        # Test that it's using Supabase storage
        if hasattr(brain, 'use_supabase') and brain.use_supabase:
            print("✓ Brain is configured to use Supabase storage")
        else:
            print("✗ Brain is not using Supabase storage")
            return False
        
        # Note: We don't test the full answer() method here as it would
        # make actual API calls and web searches
        
        return True
        
    except Exception as e:
        print(f"✗ Supabase-integrated Brain test failed: {e}")
        return False

def test_app_integration():
    """Test app.py Brain integration"""
    print("Testing app.py Brain integration...")
    
    try:
        # Import the get_user_brain function from app
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        
        # Check if API key is available
        api_key = os.environ.get('MISTRAL_API_KEY')
        if not api_key:
            print("⚠ MISTRAL_API_KEY not found, skipping app integration test")
            return True
        
        # Test that we can import the Brain class and create an instance
        from LLM import Brain
        
        # Create brain instance like the app does
        brain = Brain(
            model_name="ministral-8b-latest",
            api_key=api_key,
            use_supabase=True
        )
        print("✓ Global Brain instance created successfully for app integration")
        
        # Test that brain has the required methods
        if hasattr(brain, 'answer'):
            print("✓ Brain has answer method")
        else:
            print("✗ Brain missing answer method")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ App integration test failed: {e}")
        return False

def check_environment():
    """Check environment variables"""
    print("Checking environment variables...")
    
    required_vars = ['SUPABASE_URL', 'SUPABASE_ANON_KEY']
    optional_vars = ['MISTRAL_API_KEY']
    
    missing_required = []
    for var in required_vars:
        if not os.environ.get(var):
            missing_required.append(var)
    
    if missing_required:
        print(f"✗ Missing required environment variables: {', '.join(missing_required)}")
        return False
    else:
        print("✓ Required environment variables found")
    
    missing_optional = []
    for var in optional_vars:
        if not os.environ.get(var):
            missing_optional.append(var)
    
    if missing_optional:
        print(f"⚠ Missing optional environment variables: {', '.join(missing_optional)}")
        print("  Some tests will be skipped")
    
    return True

def main():
    """Run all tests"""
    print("Supabase Integration Test Suite")
    print("=" * 40)
    
    tests = [
        ("Environment Check", check_environment),
        ("Supabase Client", test_supabase_client),
        ("Database Setup", test_database_setup),
        ("Topic Node", test_topic_node),
        ("Chat History", test_chat_history),
        ("Brain Integration", test_brain_integration),
        ("App Integration", test_app_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        
        try:
            if test_func():
                passed += 1
            else:
                print(f"Test '{test_name}' failed")
        except Exception as e:
            print(f"Test '{test_name}' crashed: {e}")
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Supabase integration is working correctly.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)