"""
Simple code runner tool that generates code and saves output to files.
"""

import os
import sys
import json
import re
import subprocess
from datetime import datetime

class SimpleCodeGenerator:
    """A simplified code generator that saves output to files."""
    
    def __init__(self, brain, output_dir="out"):
        """Initialize the code generator.
        
        Args:
            brain: The Brain instance used for LLM generation
            output_dir: Directory to save output files
        """
        self.brain = brain
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_code(self, task_description, language="python"):
        """Generate code for a task and save to files.
        
        Args:
            task_description: Description of what the code should do
            language: Programming language to use
            
        Returns:
            Dictionary with paths to generated files
        """
        print(f"Generating code for: {task_description}")
        
        # Generate a unique ID for this task
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_id = f"task_{timestamp}"
        
        # Step 1: Generate pseudocode
        print("Step 1: Generating pseudocode...")
        pseudocode = self._generate_pseudocode(task_description)
        pseudocode_file = os.path.join(self.output_dir, f"{task_id}_pseudocode.txt")
        with open(pseudocode_file, "w") as f:
            f.write(pseudocode)
        print(f"Pseudocode saved to {pseudocode_file}")
        
        # Step 2: Generate implementation
        print("Step 2: Generating implementation...")
        implementation = self._generate_implementation(task_description, pseudocode, language)
        
        # Determine file extension
        ext = self._get_file_extension(language)
        implementation_file = os.path.join(self.output_dir, f"{task_id}_implementation.{ext}")
        with open(implementation_file, "w") as f:
            f.write(implementation)
        print(f"Implementation saved to {implementation_file}")
        
        # Step 3: Generate test cases
        print("Step 3: Generating test cases...")
        test_cases = self._generate_test_cases(task_description, implementation, language)
        test_cases_file = os.path.join(self.output_dir, f"{task_id}_test_cases.json")
        with open(test_cases_file, "w") as f:
            json.dump(test_cases, f, indent=2)
        print(f"Test cases saved to {test_cases_file}")
        
        # Step 4: Generate test file
        print("Step 4: Generating test file...")
        test_file = self._generate_test_file(task_description, implementation, test_cases, language)
        test_script_file = os.path.join(self.output_dir, f"{task_id}_test.{ext}")
        with open(test_script_file, "w") as f:
            f.write(test_file)
        print(f"Test file saved to {test_script_file}")
        
        # Step 5: Run tests if python
        test_results = None
        test_results_file = None
        if language.lower() in ["python", "py"]:
            print("Step 5: Running tests...")
            try:
                result = subprocess.run(
                    [sys.executable, test_script_file],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                test_output = result.stdout
                if result.returncode != 0:
                    test_output += f"\n\nError (return code {result.returncode}):\n{result.stderr}"
                
                test_results_file = os.path.join(self.output_dir, f"{task_id}_test_results.txt")
                with open(test_results_file, "w") as f:
                    f.write(test_output)
                print(f"Test results saved to {test_results_file}")
                
                test_results = test_output
                
                # If tests fail, refine implementation
                if "FAILED" in test_output or result.returncode != 0:
                    print("Tests failed, refining implementation...")
                    refined_implementation = self._refine_implementation(
                        task_description, implementation, test_cases, test_output, language
                    )
                    
                    # Save refined implementation
                    refined_file = os.path.join(self.output_dir, f"{task_id}_refined.{ext}")
                    with open(refined_file, "w") as f:
                        f.write(refined_implementation)
                    print(f"Refined implementation saved to {refined_file}")
                    
                    # Run tests again with refined implementation
                    print("Running tests with refined implementation...")
                    with open(implementation_file, "w") as f:
                        f.write(refined_implementation)
                    
                    # Run tests again
                    result = subprocess.run(
                        [sys.executable, test_script_file],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    refined_output = result.stdout
                    if result.returncode != 0:
                        refined_output += f"\n\nError (return code {result.returncode}):\n{result.stderr}"
                    
                    refined_results_file = os.path.join(self.output_dir, f"{task_id}_refined_results.txt")
                    with open(refined_results_file, "w") as f:
                        f.write(refined_output)
                    print(f"Refined test results saved to {refined_results_file}")
                    
                    # Update test results
                    test_results = refined_output
                    test_results_file = refined_results_file
                
            except Exception as e:
                print(f"Error running tests: {e}")
                error_file = os.path.join(self.output_dir, f"{task_id}_error.txt")
                with open(error_file, "w") as f:
                    f.write(f"Error running tests: {e}")
                print(f"Error saved to {error_file}")
        
        # Return paths to all files
        return {
            "task_id": task_id,
            "pseudocode_file": pseudocode_file,
            "implementation_file": implementation_file,
            "test_cases_file": test_cases_file,
            "test_script_file": test_script_file,
            "test_results_file": test_results_file,
            "test_results": test_results,
            "all_files": os.listdir(self.output_dir)
        }
    
    def _generate_pseudocode(self, task_description):
        """Generate pseudocode for a task.
        
        Args:
            task_description: Description of the task
            
        Returns:
            Pseudocode string
        """
        prompt = f"""
        You are an expert software engineer tasked with writing pseudocode for a coding task. 
        Write pseudocode that outlines the step-by-step approach to implementing:
        
        {task_description}
        
        Your pseudocode should:
        1. Be detailed enough to serve as a roadmap for implementation
        2. Include all necessary steps and logic
        3. Use consistent indentation to indicate nesting
        4. Cover edge cases and error handling
        5. Be language-agnostic
        
        Write ONLY the pseudocode, no explanations before or after.
        """
        
        response = self.brain.gen(prompt, max_new_tokens=1000, temperature=0.2)[0]["generated_text"]
        
        # Clean up the response (remove any explanations)
        lines = response.strip().split("\n")
        pseudocode_lines = []
        for line in lines:
            # Skip empty lines at the beginning
            if not pseudocode_lines and not line.strip():
                continue
            pseudocode_lines.append(line)
        
        return "\n".join(pseudocode_lines)
    
    def _generate_implementation(self, task_description, pseudocode, language):
        """Generate code implementation based on pseudocode.
        
        Args:
            task_description: Description of the task
            pseudocode: Generated pseudocode
            language: Programming language
            
        Returns:
            Code implementation string
        """
        prompt = f"""
        You are an expert software engineer tasked with implementing code based on pseudocode.
        Write clean, efficient, well-documented code in {language} that implements this task:
        
        Task: {task_description}
        
        Pseudocode:
        ```
        {pseudocode}
        ```
        
        Your implementation should:
        1. Follow the logic in the pseudocode exactly
        2. Include appropriate error handling
        3. Be well-commented but not excessively
        4. Follow best practices for {language}
        5. Include a function/method signature that makes the code easily testable
        
        Write ONLY the code implementation, no explanations before or after.
        """
        
        response = self.brain.gen(prompt, max_new_tokens=2000, temperature=0.2)[0]["generated_text"]
        
        # Extract code from the response (remove markdown code blocks if present)
        code_block_pattern = f"```(?:{language})?\n(.*?)```"
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # If no code block found, return the whole response
        return response.strip()
    
    def _generate_test_cases(self, task_description, implementation, language):
        """Generate test cases for the implementation.
        
        Args:
            task_description: Description of the task
            implementation: Generated code
            language: Programming language
            
        Returns:
            List of test cases
        """
        prompt = f"""
        You are an expert software tester creating comprehensive test cases.
        Create test cases for the following code that implements: {task_description}
        
        Code to test:
        ```{language}
        {implementation}
        ```
        
        Create 5-8 test cases that thoroughly test the functionality, including normal cases, edge cases, 
        and potential error conditions. For each test case, provide:
        
        1. Input value(s)
        2. Expected output
        3. A brief description of what the test is checking
        
        Format your response as a JSON array where each test case is an object with:
        - "input": The input value(s)
        - "expected": The expected output
        - "description": Brief description of the test case
        
        Your response should be ONLY the valid JSON array.
        """
        
        response = self.brain.gen(prompt, max_new_tokens=1500, temperature=0.3)[0]["generated_text"]
        
        # Try to extract JSON from the response
        json_pattern = r'\[[\s\S]*\]'
        matches = re.search(json_pattern, response, re.DOTALL)
        
        if matches:
            try:
                test_cases = json.loads(matches.group(0))
                return test_cases
            except json.JSONDecodeError:
                # If JSON parsing fails, return a default test case
                return [{"input": "example_input", "expected": "example_output", "description": "Basic test case"}]
        
        # If no JSON found, return a default test case
        return [{"input": "example_input", "expected": "example_output", "description": "Basic test case"}]
    
    def _generate_test_file(self, task_description, implementation, test_cases, language):
        """Generate a test file that can run the implementation with test cases.
        
        Args:
            task_description: Description of the task
            implementation: Generated code
            test_cases: List of test cases
            language: Programming language
            
        Returns:
            Test file content
        """
        if language.lower() in ["python", "py"]:
            # For Python, generate a test file that uses unittest
            test_file = f"""#!/usr/bin/env python3
\"\"\"
Test file for: {task_description}
\"\"\"

import unittest
import json
import sys
import os
from io import StringIO

# The implementation code
{implementation}

class TestImplementation(unittest.TestCase):
    \"\"\"Test cases for the implementation.\"\"\"
    
    def setUp(self):
        \"\"\"Set up test environment.\"\"\"
        # Capture stdout for testing print output if needed
        self.stdout_backup = sys.stdout
        self.captured_output = StringIO()
        sys.stdout = self.captured_output
    
    def tearDown(self):
        \"\"\"Clean up test environment.\"\"\"
        # Restore stdout
        sys.stdout = self.stdout_backup
    
    def get_printed_output(self):
        \"\"\"Get captured stdout content.\"\"\"
        return self.captured_output.getvalue().strip()
    
    # Try to automatically determine the main function name
    def get_main_function(self):
        \"\"\"Determine the main function to test.\"\"\"
        # Look for function definitions in the implementation
        # This is a simple heuristic that might need adjustment
        function_names = []
        lines = implementation.strip().split('\\n')
        for line in lines:
            line = line.strip()
            if line.startswith('def ') and '(' in line:
                # Extract function name
                func_name = line[4:line.index('(')]
                function_names.append(func_name)
        
        # If there's only one function, that's our target
        if len(function_names) == 1:
            return globals()[function_names[0]]
        
        # If there are multiple functions, look for one that's not a helper
        for name in function_names:
            if not name.startswith('_'):
                return globals()[name]
        
        # If all else fails, return the first function
        if function_names:
            return globals()[function_names[0]]
        
        # If no functions found, raise an error
        raise ValueError("No functions found in implementation")

"""
            
            # Add test cases
            for i, test_case in enumerate(test_cases):
                input_val = json.dumps(test_case["input"])
                expected_val = json.dumps(test_case["expected"])
                description = test_case["description"]
                
                test_file += f"""
    def test_case_{i+1}(self):
        \"\"\"Test case {i+1}: {description}\"\"\"
        # Input: {input_val}
        # Expected: {expected_val}
        
        try:
            main_function = self.get_main_function()
            
            # Handle different input types
            input_val = {input_val}
            if isinstance(input_val, list):
                actual = main_function(*input_val)
            elif isinstance(input_val, dict):
                actual = main_function(**input_val)
            else:
                actual = main_function(input_val)
            
            expected = {expected_val}
            
            # For comparing complex objects like lists
            if isinstance(expected, list) and isinstance(actual, list):
                self.assertEqual(len(expected), len(actual))
                for i, item in enumerate(expected):
                    self.assertEqual(item, actual[i])
            else:
                self.assertEqual(expected, actual)
            
        except Exception as e:
            self.fail(f"Test case {i+1} failed with error: {{e}}")
"""
            
            # Add main section
            test_file += """

if __name__ == '__main__':
    # Run tests
    unittest.main()
"""
            
            return test_file
        else:
            # For other languages, just return a placeholder
            return f"// Test file for {language} not implemented yet"
    
    def _refine_implementation(self, task_description, implementation, test_cases, test_output, language):
        """Refine the implementation to fix failing tests.
        
        Args:
            task_description: Description of the task
            implementation: Original implementation
            test_cases: Test cases
            test_output: Output from running tests
            language: Programming language
            
        Returns:
            Refined implementation
        """
        prompt = f"""
        You are an expert software developer tasked with fixing code that fails tests.
        The following implementation for "{task_description}" has failed some tests:
        
        ```{language}
        {implementation}
        ```
        
        Test cases:
        {json.dumps(test_cases, indent=2)}
        
        Test output showing failures:
        ```
        {test_output}
        ```
        
        Please fix the implementation so that it passes all the tests. The fixed implementation should:
        1. Maintain the same general structure and approach
        2. Fix any logical errors or bugs
        3. Work correctly for all the test cases
        4. Include comments explaining the fixes you made
        
        Return ONLY the fixed code implementation, no explanations before or after.
        """
        
        response = self.brain.gen(prompt, max_new_tokens=2000, temperature=0.2)[0]["generated_text"]
        
        # Extract code from the response
        code_block_pattern = f"```(?:{language})?\n(.*?)```"
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # If no code block found, return the whole response
        return response.strip()
    
    def _get_file_extension(self, language):
        """Get file extension for a programming language.
        
        Args:
            language: Programming language name
            
        Returns:
            File extension string
        """
        language_extensions = {
            "python": "py",
            "py": "py",
            "javascript": "js",
            "js": "js",
            "typescript": "ts",
            "ts": "ts",
            "java": "java",
            "c": "c",
            "cpp": "cpp",
            "c++": "cpp",
            "csharp": "cs",
            "c#": "cs",
            "go": "go",
            "rust": "rs",
            "ruby": "rb",
            "php": "php"
        }
        
        return language_extensions.get(language.lower(), "txt")

# Simple script to test the code generator
if __name__ == "__main__":
    # Import Brain here to avoid circular imports
    from LLM import Brain
    
    # Create Brain instance
    brain = Brain()
    
    # Create code generator
    generator = SimpleCodeGenerator(brain)
    
    # Generate code for a simple task
    result = generator.generate_code("Write a function to check if a string is a palindrome")
    
    # Print results
    print("\nGeneration complete!")
    print(f"Task ID: {result['task_id']}")
    print(f"All files: {', '.join(result['all_files'])}")
    
    # If test results are available, print them
    if result['test_results']:
        print("\nTest Results:")
        print(result['test_results'])