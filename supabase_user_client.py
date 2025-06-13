import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

class SupabaseUserClient:
    """
    User-authenticated Supabase client that works with Row Level Security (RLS).
    This client sets the auth token for the user to work with RLS policies.
    """
    
    def __init__(self, user_token: str = None):
        """
        Initialize a user-authenticated Supabase client.
        
        Args:
            user_token: JWT token for the authenticated user
        """
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_ANON_KEY")
        
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in environment variables")
        
        self._client = create_client(url, key)
        
        # Set the auth token if provided
        if user_token:
            self._client.auth.set_session(user_token)
    
    @property
    def client(self) -> Client:
        return self._client
    
    def set_auth_token(self, token: str):
        """Set the authentication token for RLS"""
        self._client.auth.set_session(token)

def create_user_client(user_token: str = None) -> SupabaseUserClient:
    """
    Create a user-authenticated Supabase client.
    
    Args:
        user_token: JWT token for the authenticated user
        
    Returns:
        SupabaseUserClient instance
    """
    return SupabaseUserClient(user_token)