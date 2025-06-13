class MemorySystem:
    def __init__(self):
        self.short_term_memory = []
        self.long_term_memory = {}

    def store_conversation(self, conversation_id, conversation):
        """Store a conversation in long-term memory."""
        self.long_term_memory[conversation_id] = conversation

    def retrieve_conversation(self, conversation_id):
        """Retrieve a conversation from long-term memory."""
        return self.long_term_memory.get(conversation_id, None)

    def add_to_short_term_memory(self, message):
        """Add a message to short-term memory."""
        self.short_term_memory.append(message)

    def clear_short_term_memory(self):
        """Clear short-term memory."""
        self.short_term_memory = []

    def get_short_term_memory(self):
        """Retrieve short-term memory."""
        return self.short_term_memory.copy()