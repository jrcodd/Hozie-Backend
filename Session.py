from LLM import Brain
class Session:
    """
    A class to manage a user session.
    Right now the simple approach is to just store messages in the session. Start a new session when the user logs in. 

    In the future hozie that is less of a chat bot for everyone to log into and more of a personal assistant (I might make a new repo or a fork or something for this and stop developing the front end website) I am thinking that we just have a seperate way to store everything like a new memory tree but for personal things like preferences or other personal info about people that would be weird to be in the memory tree. Like more connections type data instead of rigid facts.

    I'm thinking like this: Imagine rain falling into puddles. Each drop radiates out and affects the whole puddle. Now instead of water, imagine a bunch of different colors. Each new color leaves a mark on the world (big puddle). This is how I imagine connections. When you meet someone, you change their puddle area while also changing your own and maybe you change one of their friends too that is closeby. Over time, your colors mix together and your persona eventually becomes a product of all the other people you interact with. 
    """
    def __init__(self, debug=True):
        self.brain = Brain(debug)
        self.messages = []

    def answer(self, query: str, stream: bool = False) -> str:
        """
        Ask the llm a question and add it to the history
        
        Args:
            query (str): The question to ask the user
        """
        answer =  self.brain.answer(query, self.get_messages(5), 2, stream=stream)
        self.add_message([query, answer])
        return answer

    def add_message(self, message: list[str, str]) -> None:
        """
        Add a message to the session.

        Args:
            message (List[str, str]): A list containing the sender and the message.
        """
        self.messages.append(message)

    def get_messages(self, num_messages: int) -> list:
        """
        Get all messages in the session.
        
        Args:
            num_messages (int): The number of messages to retrieve.

        Returns:
            list: A list of messages in the session.
        """
        if num_messages > len(self.messages):
            return self.messages
        return self.messages[-num_messages:]
    
    def clear(self) -> None:
        """
        Clear the session data.
        """
        self.messages = []