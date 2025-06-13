"""
Code Generator Tool for complex coding tasks. MAJOR WIP DOES NOT WORK YET

This module provides utilities to break down complex programming tasks into 
manageable steps, generate code, and verify its correctness.
"""

import json
import os
import re
import subprocess
import textwrap
import tempfile
from typing import Dict, List, Optional, Tuple, Any

class CodingTask:
    """Represents a single coding task to be completed."""
    
    def __init__(self, 
                 task_id: str, 
                 description: str, 
                 parent_id: Optional[str] = None, 
                 code: Optional[str] = None,
                 status: str = "pending"):
        """Initialize a coding task.
        
        Args:
            task_id: Unique identifier for this task
            description: Detailed description of the task
            parent_id: Optional ID of parent task (for subtasks)
            code: Optional code that has been generated for this task
            status: Current status of the task (pending, in_progress, completed, failed)
        """
        self.task_id = task_id
        self.description = description
        self.parent_id = parent_id
        self.code = code
        self.status = status
        self.test_cases = []
        self.notes = []
        self.subtasks = []
        self.metadata = {}
    
    def add_test_case(self, input_data: Any, expected_output: Any, description: str = "") -> None:
        """Add a test case for this coding task.
        
        Args:
            input_data: Input to the function/code
            expected_output: Expected output from the function/code
            description: Optional description of what this test case verifies
        """
        self.test_cases.append({
            "input": input_data,
            "expected": expected_output,
            "description": description
        })
    
    def add_note(self, note: str) -> None:
        """Add a note or comment about the task.
        
        Args:
            note: The note to add
        """
        self.notes.append({
            "content": note,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        })
    
    def add_subtask(self, description: str) -> 'CodingTask':
        """Create and add a subtask.
        
        Args:
            description: Description of the subtask
            
        Returns:
            The newly created subtask
        """
        subtask_id = f"{self.task_id}_{len(self.subtasks) + 1}"
        subtask = CodingTask(subtask_id, description, parent_id=self.task_id)
        self.subtasks.append(subtask)
        return subtask
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary format.
        
        Returns:
            Dictionary representation of the task
        """
        return {
            "task_id": self.task_id,
            "description": self.description,
            "parent_id": self.parent_id,
            "code": self.code,
            "status": self.status,
            "test_cases": self.test_cases,
            "notes": self.notes,
            "subtasks": [subtask.to_dict() for subtask in self.subtasks],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodingTask':
        """Create a task from dictionary format.
        
        Args:
            data: Dictionary representation of the task
            
        Returns:
            Reconstructed CodingTask instance
        """
        task = cls(
            task_id=data["task_id"],
            description=data["description"],
            parent_id=data.get("parent_id"),
            code=data.get("code"),
            status=data.get("status", "pending")
        )
        
        task.test_cases = data.get("test_cases", [])
        task.notes = data.get("notes", [])
        task.metadata = data.get("metadata", {})
        
        # Recursively build subtasks
        for subtask_data in data.get("subtasks", []):
            task.subtasks.append(cls.from_dict(subtask_data))
            
        return task


class CodeGenerator:
    """Tool for generating code for complex tasks using a step-by-step approach."""
    
    def __init__(self, llm_client: Any, workspace_dir: Optional[str] = None):
        """Initialize the code generator.
        
        Args:
            llm_client: Client for interacting with the language model
            workspace_dir: Optional directory for saving temporary files and results
        """
        self.llm_client = llm_client
        self.workspace_dir = workspace_dir or tempfile.mkdtemp(prefix="hozie_coding_")
        os.makedirs(self.workspace_dir, exist_ok=True)
        
        # Track all tasks
        self.tasks = []
        self.task_map = {}  # Map task_id to task for quick lookup
    
    def create_main_task(self, description: str) -> CodingTask:
        """Create a new main (top-level) coding task.
        
        Args:
            description: Detailed description of the task
            
        Returns:
            The newly created task
        """
        task_id = f"task_{len(self.tasks) + 1}"
        task = CodingTask(task_id, description)
        self.tasks.append(task)
        self.task_map[task_id] = task
        return task
    
    def break_down_task(self, task: CodingTask) -> List[CodingTask]:
        """Break down a complex task into smaller subtasks using the LLM.
        
        Args:
            task: The task to break down
            
        Returns:
            List of created subtasks
        """
        prompt = textwrap.dedent(f"""
        You are an expert software developer tasked with breaking down a complex coding task 
        into smaller, manageable subtasks. Each subtask should be a discrete unit of work
        that can be implemented and tested independently.
        
        The task to break down is:
        {task.description}
        
        Analyze the task and break it down into 3-7 subtasks. For each subtask:
        1. Provide a clear, specific description
        2. Make sure it's small enough to be implemented in a single function or small class
        3. Ensure all subtasks together will fulfill the requirements of the parent task
        4. Order them in a logical sequence of implementation
        
        Return your answer as a JSON array with each item having:
        - "subtask_description": A detailed description of what needs to be done
        - "estimated_complexity": A number from 1-5 indicating complexity (1=simplest, 5=most complex)
        
        Your response should be ONLY the valid JSON array.
        """)
        
        # Query LLM
        response = self.llm_client.generate(prompt=prompt, temperature=0.2)
        
        # Parse response as JSON
        try:
            # Extract JSON array from response 
            json_match = re.search(r'\[[\s\S]*?\]', response)
            if json_match:
                json_str = json_match.group(0)
                subtask_data = json.loads(json_str)
                
                # Create subtasks
                subtasks = []
                for idx, data in enumerate(subtask_data, 1):
                    subtask = task.add_subtask(data["subtask_description"])
                    subtask.metadata["estimated_complexity"] = data.get("estimated_complexity", 3)
                    self.task_map[subtask.task_id] = subtask
                    subtasks.append(subtask)
                
                return subtasks
            else:
                # Fallback parsing if JSON extraction fails
                task.add_note("Failed to parse LLM subtask breakdown as JSON. Using fallback method.")
                # Extract lines that look like tasks
                lines = response.split('\n')
                subtasks = []
                for idx, line in enumerate(lines, 1):
                    # Look for numbered items, dashes, etc.
                    if re.match(r'^\s*(\d+|[-*•])', line):
                        # Clean up the line to get just the description
                        description = re.sub(r'^\s*(\d+|[-*•])\s*', '', line).strip()
                        if description:
                            subtask = task.add_subtask(description)
                            self.task_map[subtask.task_id] = subtask
                            subtasks.append(subtask)
                
                return subtasks
                
        except Exception as e:
            task.add_note(f"Error breaking down task: {e}")
            return []
    
    def generate_pseudocode(self, task: CodingTask) -> str:
        """Generate pseudocode for a task.
        
        Args:
            task: The task to generate pseudocode for
            
        Returns:
            The generated pseudocode
        """
        prompt = textwrap.dedent(f"""
        You are an expert software developer tasked with writing pseudocode for a coding task.
        Write clear pseudocode that outlines the implementation of the following task:
        
        {task.description}
        
        Your pseudocode should:
        1. Be detailed enough to serve as a roadmap for implementation
        2. Include all necessary steps and logic
        3. Use consistent indentation to indicate nesting
        4. Include comments explaining any complex parts
        
        Write ONLY the pseudocode, no explanations or introductions.
        """)
        
        # Query LLM
        response = self.llm_client.generate(prompt=prompt, temperature=0.2)
        
        # Store pseudocode in task notes
        task.add_note(f"Pseudocode:\n{response}")
        
        return response
    
    def generate_code(self, task: CodingTask, language: str, include_pseudocode: bool = True) -> str:
        """Generate actual code for a task.
        
        Args:
            task: The task to generate code for
            language: The programming language to use
            include_pseudocode: Whether to generate pseudocode first
            
        Returns:
            The generated code
        """
        # Optionally generate pseudocode first
        pseudocode = ""
        if include_pseudocode:
            pseudocode = self.generate_pseudocode(task)
        
        # Gather parent task context if this is a subtask
        parent_context = ""
        if task.parent_id and task.parent_id in self.task_map:
            parent = self.task_map[task.parent_id]
            parent_context = f"\nThis is part of a larger task: {parent.description}"
        
        # Gather sibling context
        sibling_context = ""
        if task.parent_id and task.parent_id in self.task_map:
            parent = self.task_map[task.parent_id]
            if parent.subtasks:
                sibling_descriptions = [
                    f"- {sibling.task_id}: {sibling.description}" 
                    for sibling in parent.subtasks 
                    if sibling.task_id != task.task_id
                ]
                if sibling_descriptions:
                    sibling_context = "\n\nRelated subtasks:\n" + "\n".join(sibling_descriptions)
        
        # Build the prompt
        pseudocode_section = f"\nPseudocode:\n{pseudocode}" if pseudocode else ""
        
        prompt = textwrap.dedent(f"""
        You are an expert software developer tasked with implementing code for the following task:
        
        Task: {task.description}{parent_context}{sibling_context}{pseudocode_section}
        
        Write clean, efficient, and well-documented code in {language} to implement this task.
        Your code should:
        1. Be correct and functional
        2. Include appropriate error handling
        3. Be well-commented
        4. Follow established best practices for {language}
        5. Be modular and maintainable
        
        Return ONLY the code implementation, with no explanations before or after the code.
        """)
        
        # Query LLM
        response = self.llm_client.generate(prompt=prompt, temperature=0.2)
        
        # Extract code from response
        code = self._extract_code_from_response(response, language)
        
        # Update task with generated code
        task.code = code
        task.status = "in_progress"
        
        return code
    
    def generate_test_cases(self, task: CodingTask, language: str) -> List[Dict[str, Any]]:
        """Generate test cases for a task.
        
        Args:
            task: The task to generate test cases for
            language: The programming language used
            
        Returns:
            List of generated test cases
        """
        # Ensure we have code to test
        if not task.code:
            task.add_note("Cannot generate test cases without generated code")
            return []
        
        prompt = textwrap.dedent(f"""
        You are an expert software developer tasked with creating comprehensive test cases 
        for a function or module. The code implements the following task:
        
        {task.description}
        
        Here is the code:
        ```{language}
        {task.code}
        ```
        
        Create 3-5 test cases that thoroughly test the functionality, including edge cases.
        For each test case, provide:
        1. Input values
        2. Expected output
        3. A brief description of what the test case is checking
        
        Return your answer as a JSON array where each test case is an object with:
        - "input": The input value(s) as a valid JSON value
        - "expected": The expected output as a valid JSON value
        - "description": A brief description of the test case
        
        Your response should be ONLY the valid JSON array.
        """)
        
        # Query LLM
        response = self.llm_client.generate(prompt=prompt, temperature=0.2)
        
        # Parse response as JSON
        try:
            # Extract JSON array from response
            json_match = re.search(r'\[[\s\S]*?\]', response)
            if json_match:
                json_str = json_match.group(0)
                test_cases = json.loads(json_str)
                
                # Add test cases to the task
                for test_case in test_cases:
                    task.add_test_case(
                        test_case["input"],
                        test_case["expected"],
                        test_case.get("description", "")
                    )
                
                return test_cases
            else:
                task.add_note("Failed to extract test cases as JSON from LLM response")
                return []
                
        except Exception as e:
            task.add_note(f"Error generating test cases: {e}")
            return []
    
    def run_tests(self, task: CodingTask, language: str) -> Tuple[bool, List[Dict[str, Any]]]:
        """Run tests for a task and return the results.
        
        Args:
            task: The task to test
            language: The programming language used
            
        Returns:
            Tuple of (success_flag, test_results)
        """
        if not task.code or not task.test_cases:
            task.add_note("Cannot run tests without code and test cases")
            return False, []
        
        # Generate harness code to run tests
        test_harness = self._generate_test_harness(task, language)
        
        # Write code and test harness to temporary files
        module_file = os.path.join(self.workspace_dir, f"{task.task_id}_implementation.{self._get_file_extension(language)}")
        test_file = os.path.join(self.workspace_dir, f"{task.task_id}_test.{self._get_file_extension(language)}")
        
        with open(module_file, 'w') as f:
            f.write(task.code)
            
        with open(test_file, 'w') as f:
            f.write(test_harness)
        
        # Run the tests
        try:
            test_results = self._execute_tests(test_file, language)
            
            # Update task status based on test results
            all_passed = all(result.get("passed", False) for result in test_results)
            task.status = "completed" if all_passed else "failed"
            
            # Add test results to task notes
            summary = f"Tests: {sum(1 for r in test_results if r.get('passed', False))}/{len(test_results)} passed"
            task.add_note(f"Test results: {summary}")
            
            return all_passed, test_results
            
        except Exception as e:
            task.add_note(f"Error running tests: {e}")
            task.status = "failed"
            return False, []
    
    def refine_code(self, task: CodingTask, language: str, test_results: List[Dict[str, Any]]) -> str:
        """Refine code based on test results.
        
        Args:
            task: The task to refine code for
            language: The programming language used
            test_results: Results from previous test run
            
        Returns:
            The refined code
        """
        # Filter for failed tests
        failed_tests = [r for r in test_results if not r.get("passed", False)]
        
        if not failed_tests:
            task.add_note("No failed tests to fix")
            return task.code
        
        # Format test failures for the prompt
        test_failure_str = ""
        for i, test in enumerate(failed_tests, 1):
            test_failure_str += f"\nTest {i}:\n"
            test_failure_str += f"Input: {json.dumps(test['test_case']['input'])}\n"
            test_failure_str += f"Expected: {json.dumps(test['test_case']['expected'])}\n"
            test_failure_str += f"Actual: {json.dumps(test.get('actual', 'No output'))}\n"
            test_failure_str += f"Error: {test.get('error', 'No error message')}\n"
        
        prompt = textwrap.dedent(f"""
        You are an expert software developer tasked with fixing a code implementation.
        The following code was written to fulfill this task:
        
        {task.description}
        
        Current code:
        ```{language}
        {task.code}
        ```
        
        The code failed the following tests:{test_failure_str}
        
        Fix the code to pass all tests. Do not hardcode the expected outputs.
        Make your changes as minimal as necessary to fix the issues, while maintaining code quality.
        
        Return ONLY the revised code, with no explanations before or after.
        """)
        
        # Query LLM for refined code
        response = self.llm_client.generate(prompt=prompt, temperature=0.2)
        
        # Extract code from response
        refined_code = self._extract_code_from_response(response, language)
        
        # Update task with refined code
        previous_code = task.code
        task.code = refined_code
        
        # Add refinement note
        task.add_note(f"Code refined to fix failed tests: {len(failed_tests)} test(s)")
        
        return refined_code
    
    def finalize_task(self, task: CodingTask, finalize_subtasks: bool = True) -> None:
        """Finalize a task by ensuring all tests pass and code is complete.
        
        Args:
            task: The task to finalize
            finalize_subtasks: Whether to recursively finalize subtasks
        """
        # First finalize all subtasks if requested
        if finalize_subtasks and task.subtasks:
            for subtask in task.subtasks:
                self.finalize_task(subtask, finalize_subtasks=True)
        
        # Skip if task is already completed
        if task.status == "completed":
            return
        
        # Skip if no code has been generated
        if not task.code:
            task.add_note("Cannot finalize task without generated code")
            return
        
        # If we have test cases, run them and refine code if needed
        if task.test_cases:
            language = self._detect_language(task.code)
            max_attempts = 3
            
            for attempt in range(1, max_attempts + 1):
                passed, results = self.run_tests(task, language)
                
                if passed:
                    task.status = "completed"
                    task.add_note(f"Task completed successfully on attempt {attempt}")
                    break
                    
                # Last attempt failed, try to refine the code
                if attempt < max_attempts:
                    task.add_note(f"Test run {attempt} failed. Refining code...")
                    self.refine_code(task, language, results)
                else:
                    task.add_note("Maximum refinement attempts reached. Task left as failed.")
    
    def execute_task_pipeline(self, task_description: str, language: str) -> CodingTask:
        """Execute the complete pipeline for a coding task.
        
        Args:
            task_description: Description of the task to complete
            language: Programming language to use
            
        Returns:
            The completed task
        """
        # 1. Create the main task
        main_task = self.create_main_task(task_description)
        main_task.add_note(f"Starting task pipeline in {language}")
        
        # 2. Break down into subtasks
        subtasks = self.break_down_task(main_task)
        
        if not subtasks:
            # If breakdown failed or task is simple, implement directly
            main_task.add_note("Implementing as a single task")
            
            # Generate code
            code = self.generate_code(main_task, language)
            
            # Generate test cases
            self.generate_test_cases(main_task, language)
            
            # Run tests and refine if needed
            self.finalize_task(main_task)
            
            return main_task
        
        # 3. Process each subtask
        for subtask in subtasks:
            subtask.add_note(f"Processing subtask: {subtask.description}")
            
            # Generate code for subtask
            code = self.generate_code(subtask, language)
            
            # Generate test cases for subtask
            self.generate_test_cases(subtask, language)
            
            # Finalize the subtask (run tests, refine if needed)
            self.finalize_task(subtask, finalize_subtasks=False)
        
        # 4. Assemble the final solution from subtasks
        if all(subtask.status == "completed" for subtask in subtasks):
            main_task.add_note("All subtasks completed. Assembling final solution.")
            assembled_code = self._assemble_final_solution(main_task, language)
            main_task.code = assembled_code
            
            # Generate test cases for the assembled solution
            self.generate_test_cases(main_task, language)
            
            # Run final tests
            self.finalize_task(main_task, finalize_subtasks=False)
        else:
            failed_subtasks = [s.task_id for s in subtasks if s.status != "completed"]
            main_task.add_note(f"Some subtasks failed: {failed_subtasks}")
            main_task.status = "failed"
        
        return main_task
    
    def search_web_for_help(self, task: CodingTask, query: Optional[str] = None) -> str:
        """Search the web for help with a coding task.
        
        Args:
            task: The task to get help for
            query: Optional specific query, otherwise generated from task
            
        Returns:
            Relevant information found on the web
        """
        # Generate search query if not provided
        if not query:
            prompt = textwrap.dedent(f"""
            Create a web search query to find information that would help solve this coding task:
            
            {task.description}
            
            Return only the search query, nothing else.
            """)
            
            # Query LLM for search query
            query = self.llm_client.generate(prompt=prompt, temperature=0.2).strip()
        
        # We would typically use a web search API here
        # For this implementation, we'll need to connect to an external search service
        # and process the results
        
        # Placeholder response
        task.add_note(f"Searched web for: {query}")
        return f"Would perform web search for: {query}"
    
    def save_tasks(self, filename: str) -> None:
        """Save all tasks to a JSON file.
        
        Args:
            filename: Path to the file to save to
        """
        data = [task.to_dict() for task in self.tasks]
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_tasks(self, filename: str) -> None:
        """Load tasks from a JSON file.
        
        Args:
            filename: Path to the file to load from
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.tasks = []
        self.task_map = {}
        
        for task_data in data:
            task = CodingTask.from_dict(task_data)
            self.tasks.append(task)
            
            # Rebuild task map (including all subtasks)
            def add_to_map(t):
                self.task_map[t.task_id] = t
                for st in t.subtasks:
                    add_to_map(st)
            
            add_to_map(task)
    
    # ======== Helper methods ========
    
    def _extract_code_from_response(self, response: str, language: str) -> str:
        """Extract code from an LLM response.
        
        Args:
            response: The response from the LLM
            language: The expected programming language
            
        Returns:
            The extracted code
        """
        # Try to extract code blocks with language tags
        code_block_pattern = f"```(?:{language})?\n(.*?)```"
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # Fallback: try to extract any code blocks
        generic_block_pattern = r"```\w*\n(.*?)```"
        matches = re.findall(generic_block_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # Last resort: use the whole response
        return response.strip()
    
    def _generate_test_harness(self, task: CodingTask, language: str) -> str:
        """Generate test harness code for running tests.
        
        Args:
            task: The task to generate a test harness for
            language: The programming language to use
            
        Returns:
            The generated test harness code
        """
        # Create a harness based on the language
        if language.lower() in ["python", "py"]:
            return self._generate_python_test_harness(task)
        elif language.lower() in ["javascript", "js"]:
            return self._generate_js_test_harness(task)
        elif language.lower() in ["java"]:
            return self._generate_java_test_harness(task)
        else:
            # Generic approach using the LLM
            return self._generate_generic_test_harness(task, language)
    
    def _generate_python_test_harness(self, task: CodingTask) -> str:
        """Generate a Python test harness.
        
        Args:
            task: The task to generate a test harness for
            
        Returns:
            Python test harness code
        """
        # Extract module/function name from code
        module_name = f"task_{task.task_id}_implementation"
        
        harness = [
            "import json",
            "import sys",
            "import traceback",
            f"import {module_name} # The implementation file",
            "",
            "def run_tests():",
            "    results = []",
            "    test_cases = " + json.dumps(task.test_cases, indent=4),
            "",
            "    for i, test_case in enumerate(test_cases):",
            "        result = {",
            "            'test_number': i + 1,",
            "            'test_case': test_case,",
            "            'passed': False",
            "        }",
            "",
            "        try:",
            "            # Invoke the implementation with test input",
            "            # This is a generic approach - update for specific function names",
            "            # Try to infer the main function by analyzing the module",
            "            # This is a simplified version - in practice, you'd need more sophisticated inspection",
            "            ",
            "            # Try to find callable objects defined in the module",
            "            callables = []",
            "            for name in dir(task_{0}_implementation):".format(task.task_id),
            "                if not name.startswith('_'):",
            "                    obj = getattr(task_{0}_implementation, name)".format(task.task_id),
            "                    if callable(obj):",
            "                        callables.append(name)",
            "",
            "            if not callables:",
            "                result['error'] = 'No callable functions found in implementation'",
            "                results.append(result)",
            "                continue",
            "",
            "            # Use the first callable as our test target",
            "            test_target = getattr(task_{0}_implementation, callables[0])".format(task.task_id),
            "",
            "            # Handle different input types",
            "            input_value = test_case['input']",
            "            if isinstance(input_value, list):",
            "                actual = test_target(*input_value)",
            "            elif isinstance(input_value, dict):",
            "                actual = test_target(**input_value)",
            "            else:",
            "                actual = test_target(input_value)",
            "",
            "            # Compare with expected output",
            "            expected = test_case['expected']",
            "            result['actual'] = actual",
            "",
            "            # Basic equality check - in practice, might need more sophisticated comparison",
            "            if actual == expected:",
            "                result['passed'] = True",
            "            else:",
            "                result['error'] = f'Expected {expected}, but got {actual}'",
            "",
            "        except Exception as e:",
            "            result['error'] = str(e)",
            "            result['traceback'] = traceback.format_exc()",
            "",
            "        results.append(result)",
            "",
            "    # Output results as JSON",
            "    print(json.dumps(results, indent=2))",
            "    ",
            "    # Return success flag",
            "    return all(r['passed'] for r in results)",
            "",
            "if __name__ == '__main__':",
            "    success = run_tests()",
            "    sys.exit(0 if success else 1)",
        ]
        
        return "\n".join(harness)
    
    def _generate_js_test_harness(self, task: CodingTask) -> str:
        """Generate a JavaScript test harness.
        
        Args:
            task: The task to generate a test harness for
            
        Returns:
            JavaScript test harness code
        """
        # Implementation similar to Python but for JavaScript
        module_name = f"task_{task.task_id}_implementation"
        
        harness = [
            "const fs = require('fs');",
            f"const implementation = require('./{module_name}');",
            "",
            "function runTests() {",
            "    const results = [];",
            f"    const testCases = {json.dumps(task.test_cases, indent=4)};",
            "",
            "    for (let i = 0; i < testCases.length; i++) {",
            "        const testCase = testCases[i];",
            "        const result = {",
            "            testNumber: i + 1,",
            "            testCase: testCase,",
            "            passed: false",
            "        };",
            "",
            "        try {",
            "            // Find a callable function in the implementation",
            "            const callables = [];",
            "            for (const name in implementation) {",
            "                if (typeof implementation[name] === 'function') {",
            "                    callables.push(name);",
            "                }",
            "            }",
            "",
            "            if (callables.length === 0) {",
            "                result.error = 'No callable functions found in implementation';",
            "                results.push(result);",
            "                continue;",
            "            }",
            "",
            "            // Use the first callable as our test target",
            "            const testTarget = implementation[callables[0]];",
            "",
            "            // Handle different input types",
            "            let actual;",
            "            const inputValue = testCase.input;",
            "",
            "            if (Array.isArray(inputValue)) {",
            "                actual = testTarget(...inputValue);",
            "            } else if (typeof inputValue === 'object' && inputValue !== null) {",
            "                actual = testTarget(inputValue);",
            "            } else {",
            "                actual = testTarget(inputValue);",
            "            }",
            "",
            "            // Compare with expected output",
            "            const expected = testCase.expected;",
            "            result.actual = actual;",
            "",
            "            // Basic equality check - in practice, might need more sophisticated comparison",
            "            if (JSON.stringify(actual) === JSON.stringify(expected)) {",
            "                result.passed = true;",
            "            } else {",
            "                result.error = `Expected ${JSON.stringify(expected)}, but got ${JSON.stringify(actual)}`;",
            "            }",
            "",
            "        } catch (e) {",
            "            result.error = e.message;",
            "            result.stack = e.stack;",
            "        }",
            "",
            "        results.push(result);",
            "    }",
            "",
            "    // Output results as JSON",
            "    console.log(JSON.stringify(results, null, 2));",
            "",
            "    // Return success flag",
            "    return results.every(r => r.passed);",
            "}",
            "",
            "const success = runTests();",
            "process.exit(success ? 0 : 1);",
        ]
        
        return "\n".join(harness)
    
    def _generate_java_test_harness(self, task: CodingTask) -> str:
        """Generate a Java test harness.
        
        Args:
            task: The task to generate a test harness for
            
        Returns:
            Java test harness code
        """
        # This would be a more complex implementation for Java
        # For brevity, we'll use the generic approach
        return self._generate_generic_test_harness(task, "java")
    
    def _generate_generic_test_harness(self, task: CodingTask, language: str) -> str:
        """Generate a test harness for any language using the LLM.
        
        Args:
            task: The task to generate a test harness for
            language: The programming language
            
        Returns:
            Test harness code in the specified language
        """
        prompt = textwrap.dedent(f"""
        Create a test harness in {language} to test the following implementation.
        
        Task description: {task.description}
        
        Implementation code:
        ```{language}
        {task.code}
        ```
        
        Test cases to run:
        {json.dumps(task.test_cases, indent=2)}
        
        The test harness should:
        1. Import/include the implementation code
        2. Run each test case against the implementation
        3. Compare the actual results with the expected outputs
        4. Report passes and failures
        5. Return a non-zero exit code if any tests fail
        6. Print detailed results in JSON format to stdout
        
        Return ONLY the test harness code, no explanations.
        """)
        
        # Query LLM
        response = self.llm_client.generate(prompt=prompt, temperature=0.2)
        
        # Extract code
        return self._extract_code_from_response(response, language)
    
    def _execute_tests(self, test_file: str, language: str) -> List[Dict[str, Any]]:
        """Execute tests for a specific language.
        
        Args:
            test_file: Path to the test file
            language: The programming language
            
        Returns:
            List of test results
        """
        # Generic execution function that runs the appropriate interpreter/compiler
        if language.lower() in ["python", "py"]:
            cmd = ["python", test_file]
        elif language.lower() in ["javascript", "js"]:
            cmd = ["node", test_file]
        elif language.lower() in ["java"]:
            # Compilation would be needed first
            # This is simplified
            class_name = os.path.basename(test_file).replace(".java", "")
            cmd = ["java", class_name]
        else:
            # Fallback to a generic approach using language extension
            extension = os.path.splitext(test_file)[1][1:]
            cmd = [extension, test_file]
        
        try:
            # Run the test process
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            # Parse stdout as JSON to get detailed test results
            if process.stdout:
                try:
                    return json.loads(process.stdout)
                except json.JSONDecodeError:
                    # If output is not JSON, create a basic result
                    success = process.returncode == 0
                    return [{
                        "test_number": 1,
                        "passed": success,
                        "output": process.stdout,
                        "error": process.stderr if process.stderr else None
                    }]
            
            # Return a basic result if no stdout
            return [{
                "test_number": 1,
                "passed": process.returncode == 0,
                "error": process.stderr if process.stderr else "No output from test runner"
            }]
            
        except Exception as e:
            # Return error information
            return [{
                "test_number": 1,
                "passed": False,
                "error": str(e)
            }]
    
    def _assemble_final_solution(self, main_task: CodingTask, language: str) -> str:
        """Assemble the final solution from completed subtasks.
        
        Args:
            main_task: The main task containing completed subtasks
            language: The programming language
            
        Returns:
            The assembled solution code
        """
        # Extract code from all subtasks
        subtask_codes = []
        for subtask in main_task.subtasks:
            if subtask.code:
                subtask_codes.append(f"# Subtask: {subtask.description}\n{subtask.code}")
        
        # Use LLM to assemble the solution
        subtasks_code = "\n\n".join(subtask_codes)
        
        prompt = textwrap.dedent(f"""
        You are an expert software developer tasked with assembling a complete solution from 
        multiple components.
        
        The main task is:
        {main_task.description}
        
        This task was broken down into subtasks, and each subtask has been implemented.
        Your job is to combine these implementations into a cohesive, well-structured solution.
        
        Subtask implementations:
        ```{language}
        {subtasks_code}
        ```
        
        Combine these implementations into a single, coherent solution that:
        1. Removes any redundancy or duplication
        2. Maintains a logical structure and flow
        3. Follows best practices for {language}
        4. Includes appropriate imports, class definitions, and main function
        5. Is properly documented
        
        Return ONLY the combined code, with no explanations before or after.
        """)
        
        # Query LLM
        response = self.llm_client.generate(prompt=prompt, temperature=0.2)
        
        # Extract code
        return self._extract_code_from_response(response, language)
    
    def _get_file_extension(self, language: str) -> str:
        """Get the appropriate file extension for a language.
        
        Args:
            language: The programming language
            
        Returns:
            The file extension
        """
        language_extensions = {
            "python": "py",
            "py": "py",
            "javascript": "js",
            "js": "js",
            "java": "java",
            "c": "c",
            "cpp": "cpp",
            "c++": "cpp",
            "csharp": "cs",
            "c#": "cs",
            "go": "go",
            "rust": "rs",
            "ruby": "rb",
            "php": "php",
            "typescript": "ts",
            "ts": "ts",
            "swift": "swift",
            "kotlin": "kt",
            "scala": "scala",
            "html": "html",
            "css": "css",
            "shell": "sh",
            "bash": "sh"
        }
        
        return language_extensions.get(language.lower(), "txt")
    
    def _detect_language(self, code: str) -> str:
        """Detect the programming language from code.
        
        Args:
            code: The code to analyze
            
        Returns:
            The detected language
        """
        # Simple heuristics for common languages
        if re.search(r"import\s+[a-zA-Z0-9_.]+(;|\n)|from\s+[a-zA-Z0-9_.]+\s+import", code):
            return "python"
        elif re.search(r"(const|let|var|function)\s+[a-zA-Z0-9_]+", code) or "() =>" in code:
            return "javascript"
        elif re.search(r"(public|private)\s+(static\s+)?(class|void|int|String)", code):
            return "java"
        elif "#include" in code and (re.search(r"<[a-zA-Z0-9_.]+>", code) or re.search(r'"[a-zA-Z0-9_.]+"', code)):
            return "cpp" if re.search(r"(class|template|namespace)\s+", code) else "c"
        else:
            # Default to Python if can't determine
            return "python"


class LLMClient:
    """Simple wrapper for LLM client to use with the CodeGenerator."""
    
    def __init__(self, llm_model):
        """Initialize the client with an LLM model."""
        self.model = llm_model
    
    def generate(self, prompt: str, temperature: float = 0.5) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: The prompt to send to the model
            temperature: Controls randomness (0-1)
            
        Returns:
            The generated text
        """
        # Call the underlying model's generation function
        # This implementation depends on the specific LLM client used
        
        # For Mistral API
        if hasattr(self.model, 'gen'):
            try:
                max_tokens = min(len(prompt) * 2, 4096)  # Reasonable output length
                response = self.model.gen(prompt, max_new_tokens=max_tokens, temperature=temperature)
                return response[0]["generated_text"]
            except Exception as e:
                print(f"Error generating text: {e}")
                return f"Error: {e}"
        
        # Fallback to a simple interface
        return self.model.generate(prompt, temperature=temperature)


# Example of how to use the code generator
if __name__ == "__main__":    
    from LLM import Brain
    
    # Initialize LLM
    brain = Brain()
    llm_client = LLMClient(brain)
    
    # Initialize code generator
    generator = CodeGenerator(llm_client)
    
    # Example task
    task_description = "Write a function that takes a list of integers and returns a new list with only the even numbers."
    
    # Execute the full pipeline
    completed_task = generator.execute_task_pipeline(task_description, "python")
    
    # Print results
    print(f"Task status: {completed_task.status}")
    print(f"Generated code:\n{completed_task.code}")
    
    # Save the results
    generator.save_tasks("coding_task_results.json")