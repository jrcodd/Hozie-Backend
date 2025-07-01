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
 
import textwrap
from typing import List
import argparse
from LLM import Brain
from Session import Session
from supabase_topic_node import SupabaseTopicNode

def _print_tree(node: SupabaseTopicNode, indent: str = "") -> None:
    """
    Pretty-print the SupabaseTopicNode hierarchy.
    
    Args:
        node (SupabaseTopicNode): The root node of the tree to print.
        indent (str): Current indentation level for pretty-printing.
    """
    print(f"{indent}- {node.topic}")
    for child in node.children:
        _print_tree(child, indent + "  ")


def _search_tree(node: SupabaseTopicNode, term: str, path: str = "") -> List[str]:
    """
    Return list of paths whose titles match *term* (case-insensitive).
    
    Args:
        node (SupabaseTopicNode): The root node of the tree to search.
        term (str): The search term to look for in node titles.
        path (str): Current path in the tree, used for building full paths.

    Returns:
        List[str]: List of full paths to nodes whose titles match the search term.
    """
    matches: List[str] = []
    full_path = f"{path}/{node.topic}" if path else node.topic
    if term.lower() in node.topic.lower():
        matches.append(full_path)
    for child in node.children:
        matches.extend(_search_tree(child, term, full_path))
    return matches

def _print_help() -> None:
    """
    Print the help message with available commands.
    """
    print(textwrap.dedent(
        """
        ── Available commands ────────────────────────────────────────────────
          tree                 Show the memory tree
          search <term>        Search memory titles containing <term>
          ask <question>       Ask the LLM a question
          data {name of node}  Show data for a specific node
          leafinfo {/first/second/third/.../leaf}  Show data for a specific leaf node
          think                Ask the LLM to think about a topic on its own
          merge [threshold]    Find and merge similar nodes (optional threshold 0.0-1.0, default 0.8)
          merge --live [threshold]  Actually perform the merge (not just dry run)
          quit / exit          Exit the program
        """.rstrip()
    ))


def interactive_mode(brain: Brain) -> None:
    """
    Run Hozie in interactive CLI mode.

    Args:
        brain (Brain): The Brain instance to interact with.
    """
    print("Hozie CLI — press Enter for commands. Type 'exit' to quit.")

    try:
        session = Session()
        while True:
            
            try:
                raw = input("hozie> ").strip()
            except EOFError:
                print()  
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
                reply = session.answer(question)  
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

            else:
                print(f"Unknown command: {cmd}")
                _print_help()
            

    except KeyboardInterrupt:
        print("\n^C — exiting...")


def main() -> None:
    """
    Entry point for the CLI with command-line argument support.
    """
    session = Session()
    brain = Brain(debug=True)  # Initialize the Brain with debug mode enabled
    parser = argparse.ArgumentParser(description="Hozie Voice Assistant Memory System CLI")
    parser.add_argument("--question", type=str, 
                        help="Direct question to ask without entering interactive mode")
    
    args = parser.parse_args()
    
    if args.question:
        reply = session.answer(args.question)
        print("\n>>>", reply)
    else:
        interactive_mode(brain)


if __name__ == "__main__":
    main()
