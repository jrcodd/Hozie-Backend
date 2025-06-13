"""
Task Splitter Tool for breaking down complex coding tasks.

This module provides utilities to break down complex programming tasks into
smaller, manageable steps and helps orchestrate their implementation.
"""

import json
import os
from typing import Dict, List, Optional, Any, Union
from enum import Enum


class TaskStatus(Enum):
    """Enum representing the status of a task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class CodingSubTask:
    """Represents a single step in a larger coding task."""
    
    def __init__(self, 
                 task_id: str, 
                 description: str,
                 code_comments: Optional[str] = None,
                 pseudo_code: Optional[str] = None,
                 implementation: Optional[str] = None,
                 status: TaskStatus = TaskStatus.PENDING):
        """Initialize a coding subtask.
        
        Args:
            task_id: Unique identifier for this subtask
            description: Detailed description of what this subtask should do
            code_comments: Optional detailed comments explaining the implementation approach
            pseudo_code: Optional pseudo-code outlining the implementation
            implementation: Optional actual code implementation
            status: Current status of the subtask
        """
        self.task_id = task_id
        self.description = description
        self.code_comments = code_comments
        self.pseudo_code = pseudo_code
        self.implementation = implementation
        self.status = status
        self.test_cases = []
        self.notes = []
    
    def add_test_case(self, 
                      inputs: Any, 
                      expected_output: Any, 
                      description: Optional[str] = None) -> None:
        """Add a test case for this subtask.
        
        Args:
            inputs: Input values for the test
            expected_output: Expected output for the test
            description: Optional description of what this test case checks
        """
        self.test_cases.append({
            "inputs": inputs,
            "expected_output": expected_output,
            "description": description
        })
    
    def add_note(self, content: str) -> None:
        """Add a note or observation about this subtask.
        
        Args:
            content: The note content
        """
        self.notes.append({
            "content": content,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the subtask to a dictionary.
        
        Returns:
            Dictionary representation of the subtask
        """
        return {
            "task_id": self.task_id,
            "description": self.description,
            "code_comments": self.code_comments,
            "pseudo_code": self.pseudo_code,
            "implementation": self.implementation,
            "status": self.status.value,
            "test_cases": self.test_cases,
            "notes": self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodingSubTask':
        """Create a subtask from a dictionary.
        
        Args:
            data: Dictionary representation of the subtask
            
        Returns:
            A new CodingSubTask instance
        """
        status = TaskStatus(data.get("status", "pending"))
        
        subtask = cls(
            task_id=data["task_id"],
            description=data["description"],
            code_comments=data.get("code_comments"),
            pseudo_code=data.get("pseudo_code"),
            implementation=data.get("implementation"),
            status=status
        )
        
        subtask.test_cases = data.get("test_cases", [])
        subtask.notes = data.get("notes", [])
        
        return subtask


class ComplexTask:
    """Represents a complex coding task that's broken down into subtasks."""
    
    def __init__(self, 
                 task_id: str, 
                 title: str, 
                 description: str,
                 language: str = "python"):
        """Initialize a complex coding task.
        
        Args:
            task_id: Unique identifier for this task
            title: Short title for the task
            description: Detailed description of the task requirements
            language: Programming language for this task
        """
        self.task_id = task_id
        self.title = title
        self.description = description
        self.language = language
        self.subtasks = []
        self.notes = []
        self.status = TaskStatus.PENDING
        self.final_implementation = None
    
    def add_subtask(self, description: str) -> CodingSubTask:
        """Add a new subtask to this complex task.
        
        Args:
            description: Description of the subtask
            
        Returns:
            The newly created subtask
        """
        task_id = f"{self.task_id}_subtask_{len(self.subtasks) + 1}"
        subtask = CodingSubTask(task_id, description)
        self.subtasks.append(subtask)
        return subtask
    
    def add_note(self, content: str) -> None:
        """Add a note to this complex task.
        
        Args:
            content: The note content
        """
        self.notes.append({
            "content": content,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        })
    
    def get_subtask(self, task_id: str) -> Optional[CodingSubTask]:
        """Get a subtask by its ID.
        
        Args:
            task_id: The ID of the subtask to find
            
        Returns:
            The subtask if found, None otherwise
        """
        for subtask in self.subtasks:
            if subtask.task_id == task_id:
                return subtask
        return None
    
    def update_status(self) -> None:
        """Update the overall task status based on subtask statuses."""
        if not self.subtasks:
            self.status = TaskStatus.PENDING
            return
        
        # Check if all subtasks are completed
        if all(subtask.status == TaskStatus.COMPLETED for subtask in self.subtasks):
            self.status = TaskStatus.COMPLETED
        # Check if any subtask has failed
        elif any(subtask.status == TaskStatus.FAILED for subtask in self.subtasks):
            self.status = TaskStatus.FAILED
        # Check if any subtask is in progress
        elif any(subtask.status == TaskStatus.IN_PROGRESS for subtask in self.subtasks):
            self.status = TaskStatus.IN_PROGRESS
        else:
            self.status = TaskStatus.PENDING
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the complex task to a dictionary.
        
        Returns:
            Dictionary representation of the complex task
        """
        return {
            "task_id": self.task_id,
            "title": self.title,
            "description": self.description,
            "language": self.language,
            "status": self.status.value,
            "subtasks": [subtask.to_dict() for subtask in self.subtasks],
            "notes": self.notes,
            "final_implementation": self.final_implementation
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComplexTask':
        """Create a complex task from a dictionary.
        
        Args:
            data: Dictionary representation of the complex task
            
        Returns:
            A new ComplexTask instance
        """
        task = cls(
            task_id=data["task_id"],
            title=data["title"],
            description=data["description"],
            language=data.get("language", "python")
        )
        
        task.status = TaskStatus(data.get("status", "pending"))
        task.notes = data.get("notes", [])
        task.final_implementation = data.get("final_implementation")
        
        # Create subtasks
        for subtask_data in data.get("subtasks", []):
            task.subtasks.append(CodingSubTask.from_dict(subtask_data))
        
        return task


class TaskSplitter:
    """Tool for splitting complex coding tasks into smaller, manageable subtasks."""
    
    def __init__(self, llm_client: Any):
        """Initialize the task splitter.
        
        Args:
            llm_client: Client for interacting with the language model
        """
        self.llm_client = llm_client
    
    def create_complex_task(self, 
                           title: str, 
                           description: str, 
                           language: str = "python") -> ComplexTask:
        """Create a new complex task.
        
        Args:
            title: Short title for the task
            description: Detailed description of the task requirements
            language: Programming language for this task
            
        Returns:
            A new ComplexTask instance
        """
        task_id = self._generate_task_id(title)
        return ComplexTask(task_id, title, description, language)
    
    def break_down_task(self, complex_task: ComplexTask) -> List[CodingSubTask]:
        """Break down a complex task into smaller subtasks.
        
        Args:
            complex_task: The complex task to break down
            
        Returns:
            List of created subtasks
        """
        prompt = f"""
        You are an expert software engineer tasked with breaking down a complex coding task
        into smaller, manageable subtasks.
        
        The complex task is:
        Title: {complex_task.title}
        Description: {complex_task.description}
        Programming Language: {complex_task.language}
        
        Break this task down into 3-7 logical subtasks. For each subtask:
        1. Provide a clear, specific description of what needs to be implemented
        2. Make sure each subtask is focused on a single aspect or component
        3. Order subtasks in a logical implementation sequence
        
        Format your response as a JSON array with each element being an object with:
        {{"description": "Detailed description of the subtask"}}
        
        Only respond with the JSON array, no other text.
        """
        
        # Generate subtasks using LLM
        response = self.llm_client.generate(prompt, temperature=0.3)
        
        # Parse the response to extract subtasks
        try:
            # Find JSON array in response
            import re
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                subtasks_data = json.loads(json_match.group(0))
                
                # Create subtasks
                subtasks = []
                for data in subtasks_data:
                    if "description" in data:
                        subtask = complex_task.add_subtask(data["description"])
                        subtasks.append(subtask)
                
                complex_task.add_note(f"Successfully broke down task into {len(subtasks)} subtasks")
                return subtasks
            else:
                # Fallback parsing if JSON extraction fails
                complex_task.add_note("Failed to parse LLM response as JSON. Using fallback parsing.")
                lines = response.split('\n')
                descriptions = []
                
                for line in lines:
                    # Look for lines that seem like task descriptions
                    # Check for lines starting with numbers, dashes, etc.
                    if re.match(r'^\s*(\d+[.:]|[-*•])', line):
                        # Clean up the line
                        description = re.sub(r'^\s*(\d+[.:]|[-*•])\s*', '', line).strip()
                        if description:
                            descriptions.append(description)
                
                # Create subtasks from descriptions
                subtasks = []
                for description in descriptions:
                    subtask = complex_task.add_subtask(description)
                    subtasks.append(subtask)
                
                complex_task.add_note(f"Created {len(subtasks)} subtasks using fallback parsing")
                return subtasks
                
        except Exception as e:
            complex_task.add_note(f"Error breaking down task: {str(e)}")
            return []
    
    def generate_pseudo_code(self, subtask: CodingSubTask, complex_task: ComplexTask) -> str:
        """Generate pseudo-code for a subtask.
        
        Args:
            subtask: The subtask to generate pseudo-code for
            complex_task: The parent complex task
            
        Returns:
            The generated pseudo-code
        """
        # Include parent task context and sibling information
        sibling_info = "\n".join([
            f"- {s.task_id}: {s.description}" 
            for s in complex_task.subtasks 
            if s.task_id != subtask.task_id
        ])
        
        prompt = f"""
        You are an expert software engineer tasked with writing pseudocode for a specific subtask.
        This subtask is part of a larger coding project.
        
        The overall task is:
        {complex_task.title}
        {complex_task.description}
        
        Other subtasks in this project:
        {sibling_info}
        
        Your specific subtask is:
        {subtask.description}
        
        Write detailed pseudocode that:
        1. Clearly outlines the implementation approach
        2. Includes all necessary steps and logic
        3. Accounts for edge cases and error handling
        4. Uses consistent indentation to show nesting
        5. Includes helpful comments
        
        Programming Language: {complex_task.language}
        
        Write ONLY the pseudocode, no explanations before or after.
        """
        
        # Generate pseudo-code using LLM
        pseudo_code = self.llm_client.generate(prompt, temperature=0.3)
        
        # Update subtask with generated pseudo-code
        subtask.pseudo_code = pseudo_code.strip()
        subtask.add_note("Generated pseudo-code")
        
        return subtask.pseudo_code
    
    def generate_code_comments(self, subtask: CodingSubTask, complex_task: ComplexTask) -> str:
        """Generate detailed code comments for a subtask.
        
        Args:
            subtask: The subtask to generate comments for
            complex_task: The parent complex task
            
        Returns:
            The generated code comments
        """
        prompt = f"""
        You are an expert software engineer writing detailed comments before implementing code.
        This is a preparation step to ensure your implementation approach is well-thought-out.
        
        Task: {subtask.description}
        
        This is part of a larger project:
        {complex_task.title}
        {complex_task.description}
        
        Write detailed code comments that:
        1. Explain the overall approach
        2. Identify key data structures and algorithms
        3. Highlight potential edge cases and error handling
        4. Note any performance considerations
        5. Reference any related subtasks if appropriate
        
        Format your response as actual code comments in {complex_task.language}, 
        but do not include any actual code yet.
        """
        
        # Generate code comments using LLM
        comments = self.llm_client.generate(prompt, temperature=0.3)
        
        # Update subtask with generated comments
        subtask.code_comments = comments.strip()
        subtask.add_note("Generated code comments")
        
        return subtask.code_comments
    
    def implement_subtask(self, subtask: CodingSubTask, complex_task: ComplexTask) -> str:
        """Implement code for a subtask.
        
        Args:
            subtask: The subtask to implement
            complex_task: The parent complex task
            
        Returns:
            The implemented code
        """
        # Include pseudo-code and comments if available
        pseudo_code_section = f"\nPseudo-code:\n{subtask.pseudo_code}" if subtask.pseudo_code else ""
        comments_section = f"\nCode comments:\n{subtask.code_comments}" if subtask.code_comments else ""
        
        prompt = f"""
        You are an expert software engineer implementing code for a specific subtask.
        
        Task: {subtask.description}
        
        This is part of a larger project:
        {complex_task.title}
        {complex_task.description}{pseudo_code_section}{comments_section}
        
        Write clean, efficient, and well-documented code in {complex_task.language} to implement this task.
        Your code should:
        1. Be correct and functional
        2. Include appropriate error handling
        3. Be well-commented
        4. Follow established best practices for {complex_task.language}
        5. Be easy to integrate with other components
        
        Write ONLY the implementation code, no explanations before or after.
        """
        
        # Generate implementation using LLM
        implementation = self.llm_client.generate(prompt, temperature=0.3)
        
        # Extract code from the response (remove markdown code blocks if present)
        import re
        code_block_pattern = f"```(?:{complex_task.language})?\n(.*?)```"
        matches = re.findall(code_block_pattern, implementation, re.DOTALL)
        
        if matches:
            implementation = matches[0].strip()
        
        # Update subtask with implementation
        subtask.implementation = implementation
        subtask.status = TaskStatus.COMPLETED
        subtask.add_note("Implementation completed")
        
        # Update parent task status
        complex_task.update_status()
        
        return subtask.implementation
    
    def generate_test_cases(self, subtask: CodingSubTask, complex_task: ComplexTask) -> List[Dict]:
        """Generate test cases for a subtask.
        
        Args:
            subtask: The subtask to generate test cases for
            complex_task: The parent complex task
            
        Returns:
            List of generated test cases
        """
        # Ensure we have an implementation to test
        if not subtask.implementation:
            subtask.add_note("Cannot generate test cases without an implementation")
            return []
        
        prompt = f"""
        You are an expert software tester creating comprehensive test cases for a function or module.
        
        The code you need to test implements this task:
        {subtask.description}
        
        Here is the implementation:
        ```{complex_task.language}
        {subtask.implementation}
        ```
        
        Create 3-5 test cases that thoroughly test the functionality, including edge cases.
        Format your response as a JSON array where each object has:
        1. "inputs": The input value(s) as a valid JSON value
        2. "expected_output": The expected output as a valid JSON value
        3. "description": A brief description of what this test case is checking
        
        Your response should be ONLY the valid JSON array.
        """
        
        # Generate test cases using LLM
        response = self.llm_client.generate(prompt, temperature=0.3)
        
        # Parse response to extract test cases
        try:
            import re
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                test_cases = json.loads(json_match.group(0))
                
                # Add test cases to subtask
                for test_case in test_cases:
                    subtask.add_test_case(
                        test_case.get("inputs"),
                        test_case.get("expected_output"),
                        test_case.get("description")
                    )
                
                subtask.add_note(f"Generated {len(test_cases)} test cases")
                return test_cases
            else:
                subtask.add_note("Failed to parse test cases as JSON")
                return []
                
        except Exception as e:
            subtask.add_note(f"Error generating test cases: {str(e)}")
            return []
    
    def assemble_final_solution(self, complex_task: ComplexTask) -> str:
        """Assemble the final solution from all completed subtasks.
        
        Args:
            complex_task: The complex task with completed subtasks
            
        Returns:
            The assembled final solution
        """
        # Check if all subtasks are completed
        if not all(subtask.status == TaskStatus.COMPLETED for subtask in complex_task.subtasks):
            incomplete = [s.task_id for s in complex_task.subtasks if s.status != TaskStatus.COMPLETED]
            complex_task.add_note(f"Cannot assemble final solution. Incomplete subtasks: {', '.join(incomplete)}")
            return ""
        
        # Gather all implementations
        implementations = []
        for subtask in complex_task.subtasks:
            implementations.append(f"""
            # Subtask: {subtask.description}
            {subtask.implementation}
            """)
        
        all_code = "\n\n".join(implementations)
        
        prompt = f"""
        You are an expert software engineer tasked with assembling a cohesive solution from individual components.
        
        The overall task is:
        {complex_task.title}
        {complex_task.description}
        
        Below are the individual implemented components. Your job is to integrate these into a complete,
        well-structured solution:
        
        ```{complex_task.language}
        {all_code}
        ```
        
        Create a final solution that:
        1. Combines all components correctly
        2. Eliminates redundancy and duplication
        3. Maintains a logical organization and flow
        4. Adds any necessary integration code
        5. Is properly documented
        6. Follows best practices for {complex_task.language}
        
        Return ONLY the final code, with no explanations before or after.
        """
        
        # Generate final solution using LLM
        final_solution = self.llm_client.generate(prompt, temperature=0.3)
        
        # Extract code from the response
        import re
        code_block_pattern = f"```(?:{complex_task.language})?\n(.*?)```"
        matches = re.findall(code_block_pattern, final_solution, re.DOTALL)
        
        if matches:
            final_solution = matches[0].strip()
        
        # Update complex task with final solution
        complex_task.final_implementation = final_solution
        complex_task.status = TaskStatus.COMPLETED
        complex_task.add_note("Final solution assembled")
        
        return final_solution
    
    def save_complex_task(self, complex_task: ComplexTask, filename: str) -> None:
        """Save a complex task to a JSON file.
        
        Args:
            complex_task: The complex task to save
            filename: Path to the file to save to
        """
        with open(filename, 'w') as f:
            json.dump(complex_task.to_dict(), f, indent=2)
    
    def load_complex_task(self, filename: str) -> ComplexTask:
        """Load a complex task from a JSON file.
        
        Args:
            filename: Path to the file to load from
            
        Returns:
            The loaded complex task
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        
        return ComplexTask.from_dict(data)
    
    def execute_full_workflow(self, title: str, description: str, language: str = "python") -> ComplexTask:
        """Execute the full task splitting and implementation workflow.
        
        Args:
            title: Short title for the task
            description: Detailed description of the task requirements
            language: Programming language for this task
            
        Returns:
            The completed complex task
        """
        # 1. Create the complex task
        complex_task = self.create_complex_task(title, description, language)
        complex_task.add_note("Started full workflow execution")
        
        # 2. Break down into subtasks
        subtasks = self.break_down_task(complex_task)
        
        if not subtasks:
            complex_task.add_note("Failed to break down task into subtasks")
            complex_task.status = TaskStatus.FAILED
            return complex_task
        
        # 3. Process each subtask
        for subtask in subtasks:
            try:
                # Generate pseudo-code
                self.generate_pseudo_code(subtask, complex_task)
                
                # Generate code comments
                self.generate_code_comments(subtask, complex_task)
                
                # Implement the subtask
                self.implement_subtask(subtask, complex_task)
                
                # Generate test cases
                self.generate_test_cases(subtask, complex_task)
                
            except Exception as e:
                subtask.add_note(f"Error processing subtask: {str(e)}")
                subtask.status = TaskStatus.FAILED
                complex_task.update_status()
        
        # 4. Assemble final solution if all subtasks completed
        if all(subtask.status == TaskStatus.COMPLETED for subtask in complex_task.subtasks):
            self.assemble_final_solution(complex_task)
            complex_task.add_note("Full workflow completed successfully")
        else:
            complex_task.add_note("Workflow completed with some failed subtasks")
        
        return complex_task
    
    # Helper methods
    def _generate_task_id(self, title: str) -> str:
        """Generate a task ID from a title.
        
        Args:
            title: The task title
            
        Returns:
            A task ID
        """
        # Convert title to snake case and add timestamp
        import re
        import time
        
        # Convert to lowercase and replace non-alphanumeric chars with underscore
        snake_case = re.sub(r'[^a-zA-Z0-9]', '_', title.lower())
        # Remove consecutive underscores
        snake_case = re.sub(r'_+', '_', snake_case)
        # Add timestamp
        timestamp = int(time.time())
        
        return f"{snake_case}_{timestamp}"


# Example LLM client wrapper for use with the task splitter
class LLMClientWrapper:
    """Wrapper for LLM client to use with the TaskSplitter."""
    
    def __init__(self, llm_model: Any):
        """Initialize the client wrapper.
        
        Args:
            llm_model: The underlying LLM model or client
        """
        self.model = llm_model
    
    def generate(self, prompt: str, temperature: float = 0.5) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: The prompt to send to the model
            temperature: Controls randomness (0-1)
            
        Returns:
            The generated text
        """
        # For Mistral API
        if hasattr(self.model, 'gen'):
            try:
                max_tokens = min(len(prompt) * 2, 4000)  # Reasonable output length
                response = self.model.gen(prompt, max_new_tokens=max_tokens, temperature=temperature)
                return response[0]["generated_text"]
            except Exception as e:
                print(f"Error generating text: {e}")
                return f"Error: {e}"
        
        # Fallback to a simple interface
        return self.model.generate(prompt, temperature=temperature)


# Example usage of the task splitter
if __name__ == "__main__":
    import sys
    import os
    
    # Add the parent directory to the Python path
    # This makes the imports work properly regardless of how the script is run
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(os.path.dirname(current_dir))  # src directory
    hozie_dir = os.path.dirname(src_dir)  # hozie-voice-assistant directory
    sys.path.insert(0, hozie_dir)
    
    from src.memory_system.LLM import Brain
    
    # Initialize LLM
    brain = Brain()
    llm_client = LLMClientWrapper(brain)
    
    # Initialize task splitter
    splitter = TaskSplitter(llm_client)
    
    # Example task
    task_title = "Binary Search Implementation"
    task_description = "Implement a binary search algorithm that takes a sorted array and a target value, and returns the index of the target if found, or -1 if not found."
    
    # Execute full workflow
    result = splitter.execute_full_workflow(task_title, task_description, "python")
    
    # Save the results
    splitter.save_complex_task(result, "binary_search_task.json")
    
    # Show final implementation
    if result.final_implementation:
        print("Final implementation:")
        print(result.final_implementation)
    else:
        print("Failed to generate final implementation")