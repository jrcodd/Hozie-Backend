"""Command-line interface for the Hozie brain.

Run the script and type commands.  Press **Enter** with no input to
see the command list.

Available commands
------------------
  tree                Show the whole memory tree
  search <term>       Find topic nodes whose titles contain <term>
  ask <question>      Ask the language model a question
  quit / exit         Leave the program
  data {name of node}   Show data for a specific node
  leafinfo {/first/second/third/.../leaf}   Show data for a specific leaf node
"""

from __future__ import annotations
 
from datetime import datetime, timezone
import textwrap
from typing import List
import argparse
from LLM import Brain
from topic_node import TopicNode
from task_splitter import TaskSplitter, LLMClientWrapper

# ----------------------------------------------------------------------------
#  Helper functions for memory-tree display and search
# ----------------------------------------------------------------------------

def _print_tree(node: TopicNode, indent: str = "") -> None:
    """Pretty-print the TopicNode hierarchy."""
    print(f"{indent}- {node.topic}")
    for child in node.children:
        _print_tree(child, indent + "  ")


def _search_tree(node: TopicNode, term: str, path: str = "") -> List[str]:
    """Return list of paths whose titles match *term* (case-insensitive)."""
    matches: List[str] = []
    full_path = f"{path}/{node.topic}" if path else node.topic
    if term.lower() in node.topic.lower():
        matches.append(full_path)
    for child in node.children:
        matches.extend(_search_tree(child, term, full_path))
    return matches

# ----------------------------------------------------------------------------
#  CLI loop
# ----------------------------------------------------------------------------

def _print_help() -> None:
    print(textwrap.dedent(
        """
        ── Available commands ────────────────────────────────────────────────
          tree                 Show the memory tree
          search <term>        Search memory titles containing <term>
          ask <question>       Ask the LLM a question
          data {name of node}  Show data for a specific node
          leafinfo {/first/second/third/.../leaf}  Show data for a specific leaf node
          think                Ask the LLM to think about a topic on its own
          codegen <task>   Generate code for a specific task
          quit / exit          Exit the program
        """.rstrip()
    ))


def interactive_mode(model_name: str = "mistral", host: str = "http://localhost:11434") -> None:
    """Run Hozie in interactive CLI mode."""
    brain = Brain(model_name=model_name, host=host)  # Initialize the Brain with specified parameters
    
    print("Hozie CLI — press Enter for commands. Type 'exit' to quit.")

    try:
        while True:
            try:
                raw = input("hozie> ").strip()
            except EOFError:
                print()  # handle Ctrl-D
                break

            if raw == "":
                _print_help()
                continue

            cmd, *rest = raw.split(" ", 1)
            arg = rest[0] if rest else ""

            if cmd in {"quit", "exit"}:
                break

            elif cmd == "tree":
                _print_tree(brain.memory)

            elif cmd == "search":
                if not arg:
                    print("usage: search <term>")
                    continue
                hits = _search_tree(brain.memory, arg)
                if not hits:
                    print("(no matches)")
                else:
                    print("\n".join(hits))

            elif cmd == "ask":
                question = arg or input("Question: ")
                if not question:
                    print("Please provide a question.")
                    continue
                reply = brain.answer(question)  
                print("\n>>>", reply)
            
            elif cmd == "data":
                if not arg:
                    print("usage: data {name of node}")
                    continue
                node = brain.memory.find_topic(arg)
                if node:
                    print(f"Data for {node.topic}:")
                    print(node.data)
                else:
                    print(f"No data found for topic '{arg}'")

            elif cmd == "leafinfo":
                if not arg:
                    print("usage: leafinfo {/first/second/third/.../leaf}")
                    continue
                path = arg.split("/")
                node = brain.memory.find_node_by_path(path)
                if node:
                    print(f"Leaf info for {node.topic}:")
                    print(node.data)
                else:
                    print(f"No leaf found at path '{arg}'")

            elif cmd == "think":
                question = arg or input("What should I think about? ")
                if not question:
                    print("Please provide a topic to think about.")
                    continue
                reply = brain.explore_topics_autonomously(base_topics=[question])
                print("\n>>>", reply)
            elif cmd == "codegen":
                task = arg or input("What task should I generate code for? ")
                if not task:
                    print("Please provide a task description.")
                    continue
                llm_client = LLMClientWrapper(brain)
                # Initialize code generator
                splitter = TaskSplitter(llm_client)    
                task_title = datetime.now(timezone.utc).isoformat()
                task_description = task
                # Execute full workflow
                print("Executing full workflow to code your thing... (this takes a few minutes dw I'm working on it)")
                result = splitter.execute_full_workflow(task_title, task_description, "python")
                # Save the results
                splitter.save_complex_task(result, "code_task.json")
                print("Results saved to code_task.json")
                # Show final implementation
                if result.final_implementation:
                    print("Final implementation:")
                    print(result.final_implementation)
                else:
                    print("Failed to generate final implementation")


                # Execute the full pipeline

            else:
                print(f"Unknown command: {cmd}")
                _print_help()
            

    except KeyboardInterrupt:
        print("\n^C — exiting…")


def main() -> None:
    """Entry point for the CLI with command-line argument support."""
    parser = argparse.ArgumentParser(description="Hozie Voice Assistant Memory System CLI")
    parser.add_argument("--model", type=str, default="mistral", 
                        help="Model name to use for Ollama (default: mistral)")
    parser.add_argument("--host", type=str, default="http://localhost:11434",
                        help="Ollama host URL (default: http://localhost:11434)")
    parser.add_argument("--question", type=str, 
                        help="Direct question to ask without entering interactive mode")
    
    args = parser.parse_args()
    
    if args.question:
        # Direct question mode
        brain = Brain(model_name=args.model, host=args.host)
        reply = brain.answer(args.question)
        print("\n>>>", reply)
    else:
        # Interactive mode
        interactive_mode(model_name=args.model, host=args.host)


if __name__ == "__main__":
    main()
