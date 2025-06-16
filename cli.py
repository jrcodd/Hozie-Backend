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
from supabase_topic_node import SupabaseTopicNode
import json
from typing import Dict, Tuple, Set

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

def find_and_merge_similar_nodes(brain: Brain, similarity_threshold: float = 0.8, dry_run: bool = True) -> Dict[str, List[str]]:
    """
    Find and optionally merge similar knowledge nodes in the tree using LLM-based similarity detection.
    
    Args:
        brain (Brain): The Brain instance containing the knowledge tree.
        similarity_threshold (float): Threshold for considering nodes similar (0.0 to 1.0).
        dry_run (bool): If True, only identify similar nodes without merging.
    
    Returns:
        Dict[str, List[str]]: Dictionary mapping representative node topics to lists of similar node topics.
    """
    print(f"[Merge] Starting similarity analysis {'(dry run)' if dry_run else '(live merge)'}...")
    
    # Collect all nodes with data
    all_nodes = []
    
    def collect_nodes(node: SupabaseTopicNode, path: str = "") -> None:
        full_path = f"{path}/{node.topic}" if path else node.topic
        if isinstance(node.data, dict) and node.data:
            all_nodes.append((node, full_path))
        for child in node.children:
            collect_nodes(child, full_path)
    
    collect_nodes(brain.memory)
    print(f"[Merge] Found {len(all_nodes)} nodes with data to analyze")
    
    if len(all_nodes) < 2:
        print("[Merge] Not enough nodes to compare")
        return {}
    
    # Find similar nodes using LLM
    similar_groups = {}
    processed_nodes = set()
    
    for i, (node1, path1) in enumerate(all_nodes):
        if node1.node_id in processed_nodes:
            continue
            
        print(f"[Merge] Analyzing node {i+1}/{len(all_nodes)}: '{node1.topic}'")
        
        # Find similar nodes to this one
        similar_to_node1 = [path1]
        
        for j, (node2, path2) in enumerate(all_nodes[i+1:], i+1):
            if node2.node_id in processed_nodes:
                continue
                
            # Use LLM to determine similarity
            similarity_score = _calculate_node_similarity_llm(brain, node1, node2)
            
            if similarity_score >= similarity_threshold:
                print(f"[Merge] Found similar nodes: '{node1.topic}' and '{node2.topic}' (score: {similarity_score:.2f})")
                similar_to_node1.append(path2)
                processed_nodes.add(node2.node_id)
        
        if len(similar_to_node1) > 1:
            similar_groups[path1] = similar_to_node1
            processed_nodes.add(node1.node_id)
    
    print(f"[Merge] Found {len(similar_groups)} groups of similar nodes")
    
    # Print results
    for representative, similar_nodes in similar_groups.items():
        print(f"\n[Merge] Group: {representative}")
        for similar_node in similar_nodes[1:]:  # Skip the representative
            print(f"[Merge]   → Similar to: {similar_node}")
    
    # Perform actual merging if not dry run
    if not dry_run and similar_groups:
        print("\n[Merge] Starting merge process...")
        for representative_path, similar_paths in similar_groups.items():
            _merge_node_group(brain, representative_path, similar_paths[1:])  # Skip representative
        print("[Merge] Merge process completed")
    
    return similar_groups

def _calculate_node_similarity_llm(brain: Brain, node1: SupabaseTopicNode, node2: SupabaseTopicNode) -> float:
    """
    Use LLM to calculate similarity between two nodes.
    
    Args:
        brain (Brain): Brain instance for LLM calls.
        node1 (SupabaseTopicNode): First node to compare.
        node2 (SupabaseTopicNode): Second node to compare.
    
    Returns:
        float: Similarity score between 0.0 and 1.0.
    """
    # Prepare node data for comparison
    def extract_node_info(node: SupabaseTopicNode) -> str:
        info = f"Topic: {node.topic}\n"
        if isinstance(node.data, dict):
            if node.data.get("description"):
                info += f"Description: {node.data['description'][:200]}...\n"
            if node.data.get("bullet_points"):
                points = node.data["bullet_points"][:3]  # First 3 bullet points
                info += f"Key Points: {', '.join(str(p) for p in points)}\n"
        return info
    
    node1_info = extract_node_info(node1)
    node2_info = extract_node_info(node2)
    
    prompt = textwrap.dedent(f"""
        Compare these two knowledge nodes and determine how similar they are in content and topic.
        
        NODE 1:
        {node1_info}
        
        NODE 2:
        {node2_info}
        
        Return ONLY a JSON object with this structure:
        {{
            "similarity_score": 0.85,
            "reasoning": "Brief explanation of why they are similar or different"
        }}
        
        Similarity score should be:
        - 0.9-1.0: Nearly identical content
        - 0.7-0.89: Very similar topics with overlapping content
        - 0.5-0.69: Related topics with some shared concepts
        - 0.0-0.49: Different topics or minimal overlap
    """)
    
    json_schema = {
        "type": "object",
        "properties": {
            "similarity_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reasoning": {"type": "string"}
        },
        "required": ["similarity_score", "reasoning"]
    }
    
    try:
        output = brain.sync_llm_call(prompt, temp=0.3, output_type="json_object", json_schema=json_schema)
        result = json.loads(output.strip())
        score = result.get("similarity_score", 0.0)
        reasoning = result.get("reasoning", "No reasoning provided")
        
        print(f"[Merge] Similarity between '{node1.topic}' and '{node2.topic}': {score:.2f} - {reasoning}")
        return float(score)
        
    except (json.JSONDecodeError, Exception) as e:
        print(f"[Merge] Error calculating similarity: {e}")
        return 0.0

def _merge_node_group(brain: Brain, representative_path: str, similar_paths: List[str]) -> None:
    """
    Merge a group of similar nodes into the representative node.
    
    Args:
        brain (Brain): Brain instance containing the knowledge tree.
        representative_path (str): Path to the node that will be kept.
        similar_paths (List[str]): Paths to nodes that will be merged into the representative.
    """
    print(f"[Merge] Merging {len(similar_paths)} nodes into '{representative_path}'")
    
    # Find the representative node
    rep_parts = representative_path.split("/")
    rep_node = brain.memory.find_node_by_path(rep_parts)
    
    if not rep_node:
        print(f"[Merge] Error: Could not find representative node at path '{representative_path}'")
        return
    
    # Merge data from similar nodes
    for similar_path in similar_paths:
        similar_parts = similar_path.split("/")
        similar_node = brain.memory.find_node_by_path(similar_parts)
        
        if not similar_node:
            print(f"[Merge] Warning: Could not find node at path '{similar_path}'")
            continue
        
        print(f"[Merge] Merging '{similar_node.topic}' into '{rep_node.topic}'")
        
        # Merge the data
        if isinstance(similar_node.data, dict) and isinstance(rep_node.data, dict):
            # Merge descriptions
            if similar_node.data.get("description") and rep_node.data.get("description"):
                rep_node.data["description"] += f"\n\nAdditional info: {similar_node.data['description']}"
            elif similar_node.data.get("description"):
                rep_node.data["description"] = similar_node.data["description"]
            
            # Merge bullet points
            if similar_node.data.get("bullet_points") and rep_node.data.get("bullet_points"):
                existing_points = set(rep_node.data["bullet_points"])
                for point in similar_node.data["bullet_points"]:
                    if point not in existing_points:
                        rep_node.data["bullet_points"].append(point)
            elif similar_node.data.get("bullet_points"):
                rep_node.data["bullet_points"] = similar_node.data["bullet_points"]
            
            # Merge sources
            if similar_node.metadata.get("sources"):
                rep_node.metadata.setdefault("sources", []).extend(similar_node.metadata["sources"])
        
        # Remove the similar node
        if similar_node.parent:
            similar_node.parent.children.remove(similar_node)
        
        print(f"[Merge] Successfully merged and removed '{similar_node.topic}'")
    
    print(f"[Merge] Completed merging into '{rep_node.topic}'")

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

            elif cmd == "merge":
                # Parse arguments for merge command
                args_parts = arg.split() if arg else []
                dry_run = True
                threshold = 0.8
                
                # Check for --live flag
                if "--live" in args_parts:
                    dry_run = False
                    args_parts = [a for a in args_parts if a != "--live"]
                
                # Check for threshold argument
                if args_parts:
                    try:
                        threshold = float(args_parts[0])
                        if not (0.0 <= threshold <= 1.0):
                            print("Threshold must be between 0.0 and 1.0")
                            continue
                    except ValueError:
                        print("Invalid threshold value. Must be a number between 0.0 and 1.0")
                        continue
                
                print(f"Starting merge analysis with threshold {threshold} {'(live merge)' if not dry_run else '(dry run)'}")
                
                try:
                    similar_groups = find_and_merge_similar_nodes(brain, threshold, dry_run)
                    
                    if similar_groups:
                        print(f"\nFound {len(similar_groups)} groups of similar nodes.")
                        if dry_run:
                            print("Use 'merge --live' to actually perform the merge.")
                    else:
                        print("No similar nodes found with the current threshold.")
                        
                except Exception as e:
                    print(f"Error during merge operation: {e}")
           
            else:
                print(f"Unknown command: {cmd}")
                _print_help()
            

    except KeyboardInterrupt:
        print("\n^C — exiting...")


def main() -> None:
    """
    Entry point for the CLI with command-line argument support.
    """
    brain = Brain(debug=True)  # Initialize the Brain with debug mode enabled
    parser = argparse.ArgumentParser(description="Hozie Voice Assistant Memory System CLI")
    parser.add_argument("--question", type=str, 
                        help="Direct question to ask without entering interactive mode")
    
    args = parser.parse_args()
    
    if args.question:
        reply = brain.answer(args.question)
        print("\n>>>", reply)
    else:
        interactive_mode(brain)


if __name__ == "__main__":
    main()
