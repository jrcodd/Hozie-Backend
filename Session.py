from typing import Dict
from LLM import Brain
class Session:
    """
    A class to manage a user session.
    Right now the simple approach is to just store messages in the session. Start a new session when the user logs in. 

    In the future hozie that is less of a chat bot for everyone to log into and more of a personal assistant (I might make a new repo or a fork or something for this and stop developing the front end website) I am thinking that we just have a seperate way to store everything like a new memory tree but for personal things like preferences or other personal info about people that would be weird to be in the memory tree. Like more connections type data instead of rigid facts.

    I'm thinking like this: Imagine rain falling into puddles. Each drop radiates out and affects the whole puddle. Now instead of water, imagine a bunch of different colors. Each new color leaves a mark on the world (big puddle). This is how I imagine connections. When you meet someone, you change their puddle area while also changing your own and maybe you change one of their friends too that is closeby. Over time, your colors mix together and your persona eventually becomes a product of all the other people you interact with. 
    """
    def __init__(self):
        self.brain = Brain(False)
        self.messages = []

    def answer(self, query: str) -> str:
        """
        Ask the llm a question and add it to the history
        
        Args:
            query (str): The question to ask the user
        """
        return self.brain.answer(query, 5)

    def add_message(self, message: Dict[str, str]) -> None:
        """
        Add a message to the session.

        Args:
            message (Dict[str, str]): A dictionary containing the sender and the message.
        """
        self.messages.append(message)

    def get_messages(self) -> list:
        """
        Get all messages in the session.

        Returns:
            list: A list of messages in the session.
        """
        return self.messages
    
    def clear(self) -> None:
        """
        Clear the session data.
        """
        self.messages = []