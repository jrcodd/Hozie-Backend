import re
from collections import deque
from typing import List, Dict, Any

class TopicNode:
    """
    TopicNode Class with Dynamic Web Search Integration
    ==================================================

    The TopicNode class implements a tree-based memory system for AI that dynamically grows and
    populates itself through web searches. When a user asks about a topic, the AI can search for
    information, organize it hierarchically, and store it in the appropriate nodes for future reference.

    Class Structure
    --------------
    class TopicNode:
        - topic: str - The name of the topic this node represents
        - data: dict - Structured information about this topic gathered from web searches
        - children: list[TopicNode] - List of child nodes (more specific subtopics)
        - parent: TopicNode or None - Reference to parent node (None for root)
        - metadata: dict - Additional information like source URLs, timestamps, confidence scores
        - last_updated: datetime - When this node's data was last refreshed

    Core Methods
    -----------
    - __init__(topic_name: str, data=None, parent=None, metadata=None):
        Constructor that initializes a new topic node with the given name.
        'data' starts empty or with initial information if provided.
        'parent' is None by default (for the root node).
        'metadata' tracks information sources and quality metrics.

    - add_child(topic_name: str, data=None, metadata=None) -> TopicNode:
        Creates a new child node with the given topic name and initial data.

    - find_or_create_path(path: list[str]) -> TopicNode:
        Navigates through the given path of topics, creating any missing nodes along the way.
        Returns the node at the end of the path.
        Example: node.find_or_create_path(["History", "Wars", "American Revolution"])

    Web Search and Knowledge Management
    ----------------------------------
    - search_and_populate(query: str, search_depth: int = 2) -> None:
        Performs a web search on the query, extracts relevant information, and:
        1. Identifies the appropriate place in the tree for this information
        2. Creates any necessary nodes and subnodes
        3. Populates the nodes with structured data
        4. Adds metadata about sources and confidence
        
        The search_depth parameter determines how deeply to categorize the information.
        Example: For "American Revolution", depth 1 might just create/update that node,
        while depth 3 might create/update nodes for specific battles, people, impacts, etc.

    - expand_topic(expansion_level: int = 1) -> None:
        Expands knowledge about this topic by searching for more specific information
        and creating child nodes. The expansion_level determines how many levels of
        children to generate.

    - refresh_data() -> None:
        Re-searches web sources to update this node's data with the latest information.
        Updates the last_updated timestamp.

    - categorize_information(raw_data: dict) -> dict[str, dict]:
        Takes raw information from a web search and organizes it into categories
        that will become child nodes (e.g., Battles, People, Timeline, etc.).

    - merge_with_existing(new_data: dict) -> None:
        Intelligently merges new information with existing data, resolving conflicts
        by preferring higher-confidence or more recent information.

    Query and Retrieval Methods
    --------------------------
    - get_relevant_context(query: str, depth: int = 2) -> list[dict]:
        Searches this node and children up to specified depth for information relevant 
        to the query. Returns a list of data objects to provide context for answering.

    - find_information_gaps(query: str) -> list[str]:
        Analyzes a query against existing knowledge to identify what information
        is missing that would be helpful to answer the query fully.

    - suggest_related_topics(query: str) -> list[str]:
        Based on the current tree structure, suggests related topics that might
        interest a user who asked about the given query.

    Example Usage Scenario
    ---------------------
    # User asks about the American Revolution
    user_query = "Tell me about the American Revolution"

    # Check if we already have information
    history_node = knowledge_tree.find_child("History")
    if history_node:
        wars_node = history_node.find_child("Wars")
        if wars_node:
            revolution_node = wars_node.find_child("American Revolution")
            if revolution_node:
                # We have some information, but let's make sure it's comprehensive
                # by checking for information gaps related to the query
                gaps = revolution_node.find_information_gaps(user_query)
                if gaps:
                    # Search for missing information
                    for gap in gaps:
                        revolution_node.search_and_populate(gap)
                
                # Now use the information to answer
                context = revolution_node.get_relevant_context(user_query)
                # AI formulates response using context...
                return

    # If we reach here, we don't have sufficient information, so search and build the tree
    if not history_node:
        history_node = knowledge_tree.add_child("History")
    if not wars_node:
        wars_node = history_node.add_child("Wars")

    # Create the American Revolution node and populate it deeply
    revolution_node = wars_node.add_child("American Revolution")
    revolution_node.search_and_populate(user_query, search_depth=3)

    # Now the tree might contain nodes like:
    # History/Wars/American Revolution/Battles/Bunker Hill
    # History/Wars/American Revolution/People/George Washington
    # History/Wars/American Revolution/Timeline/Declaration of Independence
    # History/Wars/American Revolution/Impact/Political

    # Use the newly gathered information to answer
    context = revolution_node.get_relevant_context(user_query)
    # AI formulates response using context...

    Advanced Features
    ---------------
    - knowledge_decay(decay_rate: float) -> None:
        Implements a form of "forgetting" by reducing confidence in older information
        based on how long ago it was added. This encourages refreshing of knowledge.

    - detect_conflicting_information() -> list[tuple]:
        Identifies pieces of information within the data that contradict each other,
        returning pairs of conflicting statements with their sources.

    - prune_tree(access_threshold: datetime) -> int:
        Removes nodes that haven't been accessed since the given threshold date,
        returning the number of nodes pruned.

    - export_to_knowledge_graph() -> Graph:
        Converts the tree to a more flexible knowledge graph structure where
        cross-relationships between non-parent/child nodes can be represented.

    Implementation Notes
    -------------------
    - Consider using an async implementation for search_and_populate() to avoid
    blocking while waiting for web search results.
    - Implement rate limiting for web searches to avoid overloading search services.
    - Use NLP techniques to improve the categorization of information into the tree.
    - Consider storing vector embeddings of node content for semantic similarity searches.
    - Implement a caching mechanism for frequent searches to reduce external API calls.
    - Design the data structure to handle conflicting information from different sources.
    - For production systems, implement authentication for web searches that require it.
    - Consider privacy implications of storing search results and implement appropriate
    data retention policies.
    """
    
    def __init__(self, topic_name, data=None, parent=None, metadata=None):
        """
        Initialize a new TopicNode.
        
        Args:
            topic_name (str): The name of this topic
            data (object, optional): Data associated with this topic. Defaults to None.
            parent (TopicNode, optional): Parent node. Defaults to None.
            metadata (dict, optional): Additional metadata about this topic. Defaults to None.
        """
        self.topic = topic_name
        self.data = data if data is not None else {}
        self.children = []
        self.parent = parent
        self.metadata = metadata if metadata is not None else {}
    
    def add_child(self, topic_name, data=None, metadata=None):
        """
        Add a child node to this node.
        
        Args:
            topic_name (str): The name of the child topic
            data (object, optional): Data for the child topic. Defaults to None.
            metadata (dict, optional): Metadata for the child topic. Defaults to None.
            
        Returns:
            TopicNode: The newly created child node
        """
        child = TopicNode(topic_name, data, self, metadata)
        self.children.append(child)
        return child
    
    def find_child(self, topic_name):
        """
        Find a direct child of this node by topic name.
        
        Args:
            topic_name (str): The name of the topic to find
            
        Returns:
            TopicNode or None: The child node if found, None otherwise
        """
        for child in self.children:
            if child.topic == topic_name:
                return child
        return None
    
    def update_data(self, new_data):
        """
        Update the data associated with this topic.
        
        Args:
            new_data (object): The new data to associate with this topic
        """
        self.data = new_data
    
    def get_path(self):
        """
        Get the full path from root to this node.
        
        Returns:
            list[str]: List of topic names from root to this node
        """
        if self.parent is None:
            return [self.topic]
        else:
            return self.parent.get_path() + [self.topic]
    
    def get_depth(self):
        """
        Get the depth of this node in the tree (root has depth 0).
        
        Returns:
            int: The depth of this node
        """
        if self.parent is None:
            return 0
        else:
            return 1 + self.parent.get_depth()
    
    def remove_child(self, topic_name):
        """
        Remove a child node by topic name.
        
        Args:
            topic_name (str): The name of the child topic to remove
            
        Returns:
            bool: True if child was found and removed, False otherwise
        """
        for i, child in enumerate(self.children):
            if child.topic == topic_name:
                self.children.pop(i)
                return True
        return False
    
    def get_all_descendants(self):
        """
        Get all descendants of this node.
        
        Returns:
            list[TopicNode]: List of all descendant nodes
        """
        descendants = []
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_all_descendants())
        return descendants
    
    def get_root(self):
        """
        Get the root node of the tree.
        
        Returns:
            TopicNode: The root node
        """
        if self.parent is None:
            return self
        else:
            return self.parent.get_root()
    
    def find_topic(self, topic_name):
        """
        Find a topic in the subtree rooted at this node.
        
        Args:
            topic_name (str): The name of the topic to find
            
        Returns:
            TopicNode or None: The node if found, None otherwise
        """
        if self.topic == topic_name:
            return self
            
        # Check immediate children first (breadth-first approach)
        for child in self.children:
            if child.topic == topic_name:
                return child
                
        # Then check descendants
        for child in self.children:
            result = child.find_topic(topic_name)
            if result:
                return result
                
        return None
    
    def print_tree(self, indent=0):
        """
        Print the tree structure with indentation.
        
        Args:
            indent (int, optional): Current indentation level. Defaults to 0.
        """
        print(" " * indent + "- " + self.topic)
        for child in self.children:
            child.print_tree(indent + 2)
    
    def to_dict(self):
        """
        Convert this node and its subtree to a dictionary.
        
        Returns:
            dict: Dictionary representation of this node and its subtree
        """
        return {
            "topic": self.topic,
            "data": self.data,
            "metadata": self.metadata,
            "children": [child.to_dict() for child in self.children]
        }
    def get_relevant_context(self, query: str, depth: int = 2, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Return up to max_results data objects from this node and its descendants
        that are most relevant to the query using basic token matching.
        """
        tokens = set(re.findall(r"\w+", query.lower()))
        if not tokens:
            return []

        queue = deque([(self, 0)])
        scored_items = []  # (score, -depth, node.data)

        while queue:
            node, d = queue.popleft()
            if d > depth:
                continue

            # Assemble searchable text for this node
            data_text = ""
            if isinstance(node.data, dict):
                data_text = " ".join(map(str, node.data.values()))
            elif node.data is not None:
                data_text = str(node.data)

            blob = f"{node.topic} {data_text}".lower()

            # Simple token-overlap score
            score = sum(1 for t in tokens if t in blob)

            if score > 0 and node.data:
                scored_items.append((score, -d, node.data))

            # enqueue children
            for child in node.children:
                queue.append((child, d + 1))

        # Sort and cut results
        scored_items.sort(reverse=True)  # highest score first
        return [item[2] for item in scored_items[:max_results]]
    @staticmethod
    def from_dict(data_dict, parent=None):
        """
        Create a TopicNode tree from a dictionary representation.
        
        Args:
            data_dict (dict): Dictionary representation of a node
            parent (TopicNode, optional): Parent node. Defaults to None.
            
        Returns:
            TopicNode: The root of the constructed tree
        """
        node = TopicNode(
            data_dict["topic"], 
            data_dict.get("data"), 
            parent,
            data_dict.get("metadata")
        )
        
        for child_dict in data_dict.get("children", []):
            child = TopicNode.from_dict(child_dict, node)
            node.children.append(child)
            
        return node
    
    def analyze_structure(self):
        """
        Analyzes the current tree structure to identify areas needing reorganization.
        
        Returns:
            dict: A report containing:
                - 'deep_branches': List of paths that exceed maximum recommended depth
                - 'wide_nodes': Nodes with too many direct children
                - 'redundancy_clusters': Groups of potentially redundant nodes
                - 'isolated_nodes': Nodes with minimal connections to the rest of the tree
        
        Algorithm:
        1. Perform depth-first traversal to build full tree statistics
        2. Check each node against threshold criteria
        3. Use content similarity metrics to identify potential redundancies
        4. Return comprehensive analysis for use by reorganization methods
        
        Edge cases:
        - Empty tree: Returns empty analysis with no recommendations
        - Single node tree: No reorganization needed, returns minimal report
        - Highly interconnected content: May need additional metrics beyond structural analysis
        """
        # Implementation would go here
        analysis = {
            'deep_branches': [],
            'wide_nodes': [],
            'redundancy_clusters': [],
            'isolated_nodes': []
        }
        
        # Analyze depth (identify paths exceeding MAX_DEPTH)
        # Analyze breadth (identify nodes with children count > MAX_CHILDREN)
        # Analyze content similarity (identify potential redundant nodes)
        # Analyze connectivity (identify isolated branches)
        
        return analysis
    
    def group_children(self, grouping_criteria=None, max_children_per_group=8):
        """
        Groups children of this node into logical subcategories to reduce direct children count.
        
        Args:
            grouping_criteria (function, optional): Function that determines how to group children.
                Default groups by similarity or common prefixes if not provided.
            max_children_per_group (int, optional): Maximum children per created group. Default is 8.
            
        Returns:
            int: Number of new groups created
            
        Algorithm:
        1. If node has fewer children than threshold, do nothing and return 0
        2. Generate groups based on similarity or provided criteria:
           a. Text similarity of topic names
           b. Content similarity of data objects
           c. Existing metadata commonalities
        3. Create new intermediate nodes for each group
        4. Move appropriate children under these new group nodes
        5. Return the number of groups created
        
        Edge cases:
        - No children: Returns 0 with no changes
        - All children very similar: May create single group, defeating purpose - needs additional check
        - All children very different: May create many small groups - should adjust criteria dynamically
        - Custom grouping function fails: Falls back to default grouping method
        """
        # Implementation would go here
        
        # Check if grouping is even needed
        if len(self.children) <= max_children_per_group:
            return 0
            
        # Default grouping based on topic name similarities
        # This could use techniques like:
        # - Common prefix extraction
        # - Topic embedding similarity
        # - Metadata-based clustering
        
        # Create new intermediate nodes for groups
        
        # Move children to appropriate groups
        
        # Return count of new groups created
        return 0  # Placeholder
    
    def merge_similar_nodes(self, similarity_threshold=0.8):
        """
        Identifies and merges nodes with highly similar content throughout the subtree.
        
        Args:
            similarity_threshold (float, optional): Threshold from 0.0 to 1.0 that determines
                how similar nodes must be to be merged. Default is 0.8 (80% similar).
                
        Returns:
            list: List of merge operations performed, each containing 'primary' and 'merged' node info
            
        Algorithm:
        1. Build content embeddings/fingerprints for all nodes in the subtree
        2. Identify pairs of nodes with similarity above threshold
        3. For each similar pair:
           a. Designate one as primary (typically the more comprehensive one)
           b. Merge unique data from secondary into primary
           c. Redirect all references/paths to the secondary node to primary
           d. Remove the secondary node
        4. Return record of all merges for reference
        
        Edge cases:
        - Circular references: Detect and prevent circular merges
        - Cascading merges: Handle when merging A→B triggers B→C merge potential
        - Conflicting data: Need conflict resolution strategy when data disagrees
        - Different but related topics: Must avoid merging topics that are related but distinct
        """
        # Implementation would go here
        merge_records = []
        
        # Get all nodes in this subtree
        all_nodes = [self] + self.get_all_descendants()
        
        # Calculate similarity between all node pairs
        # This could use techniques like:
        # - Text similarity of data content
        # - Structural similarity of child structure
        # - Semantic meaning comparison
        
        # Identify merge candidates
        
        # Perform merges while handling special cases
        
        return merge_records
    
    def optimize_structure(self, max_depth=5, max_breadth=10, similarity_threshold=0.7):
        """
        Performs comprehensive tree optimization to maintain an efficient knowledge structure.
        This is the main reorganization method that coordinates all other reorganization functions.
        
        Args:
            max_depth (int, optional): Maximum recommended tree depth. Default is 5.
            max_breadth (int, optional): Maximum recommended children per node. Default is 10.
            similarity_threshold (float, optional): Threshold for merging similar nodes. Default is 0.7.
            
        Returns:
            dict: Summary of optimizations performed
            
        Algorithm:
        1. Analyze current tree structure
        2. Apply reorganizations in specific order:
           a. First merge redundant nodes to avoid unnecessary grouping
           b. Then group children of wide nodes
           c. Finally handle any deep branches through restructuring
        3. Validate tree integrity after each operation
        4. Return comprehensive report of changes made
        
        Edge cases:
        - Very large trees: May need to process in chunks
        - Conflict between optimizations: Need to resolve when one operation would undo another
        - Critical knowledge paths: Should preserve important direct relationships
        - Recently accessed nodes: May want to avoid reorganizing frequently used paths
        """
        # Implementation would go here
        optimization_report = {
            'merges_performed': 0,
            'groups_created': 0,
            'nodes_relocated': 0,
            'total_nodes_before': 0,
            'total_nodes_after': 0
        }
        
        # 1. Count nodes before optimization
        all_nodes = [self] + self.get_all_descendants()
        optimization_report['total_nodes_before'] = len(all_nodes)
        
        # 2. Analyze structure to identify problems
        analysis = self.analyze_structure()
        
        # 3. First pass: Merge similar nodes
        merged_nodes = self.merge_similar_nodes(similarity_threshold)
        optimization_report['merges_performed'] = len(merged_nodes)
        
        # 4. Second pass: Group children of wide nodes
        for node_path in analysis['wide_nodes']:
            node = self.find_node_by_path(node_path)
            groups_created = node.group_children(max_children_per_group=max_breadth)
            optimization_report['groups_created'] += groups_created
        
        # 5. Handle deep branches through restructuring
        # This might involve:
        # - Moving deeply nested popular nodes higher in the tree
        # - Creating shortcut references to deep nodes
        # - Flattening overly deep hierarchies
        
        # 6. Count nodes after optimization
        all_nodes_after = [self] + self.get_all_descendants()
        optimization_report['total_nodes_after'] = len(all_nodes_after)
        
        return optimization_report
    
    def find_node_by_path(self, path):
        """
        Finds a node by following a path of topic names from this node.
        
        Args:
            path (list): List of topic names forming a path from this node
            
        Returns:
            TopicNode or None: The node at the end of the path, or None if not found
            
        Algorithm:
        1. Start at the current node
        2. For each topic name in the path:
           a. Find the child with matching topic
           b. If no match, return None
           c. If match, continue to next topic in path
        3. Return the final node reached
        
        Edge cases:
        - Empty path: Returns self
        - Path starts with different node: Will not find target
        - Path contains non-existent nodes: Returns None
        """
        if not path:
            return self
            
        current = self
        for topic in path:
            found = False
            for child in current.children:
                if child.topic == topic:
                    current = child
                    found = True
                    break
            if not found:
                return None
                
        return current
    
    def compute_content_similarity(self, other_node):
        """
        Computes a similarity score between this node and another node based on content.
        
        Args:
            other_node (TopicNode): The node to compare with
            
        Returns:
            float: Similarity score from 0.0 to 1.0, where 1.0 means identical content
            
        Algorithm:
        1. Convert both nodes' data to comparable format (text, vectors, etc.)
        2. Compare using appropriate similarity metric:
           a. Text data: cosine similarity of TF-IDF vectors
           b. Structured data: field-by-field comparison with weighting
           c. Numeric data: normalized distance measures
        3. Return normalized similarity score
        
        Edge cases:
        - Empty data: Should handle nodes with minimal or no data
        - Different data types: Must normalize across different types of content
        - Large data objects: Consider performance for very large content comparisons
        """
        # Placeholder implementation - would be customized based on data structure
        similarity = 0.0
        
        # Example approaches:
        # 1. For dictionary data, compare keys and values
        # 2. For text data, use NLP similarity metrics
        # 3. For mixed data, compute weighted combination of similarities
        
        return similarity
    
    def relocate_node(self, node, new_parent):
        """
        Relocates a node from its current parent to a new parent.
        Useful during restructuring to move branches to more appropriate locations.
        
        Args:
            node (TopicNode): The node to relocate
            new_parent (TopicNode): The new parent for the node
            
        Returns:
            bool: True if relocation was successful, False otherwise
            
        Algorithm:
        1. Verify relocation won't create a cycle
        2. Remove node from current parent's children
        3. Add node to new parent's children
        4. Update node's parent reference
        5. Return success status
        
        Edge cases:
        - Node not in tree: Should verify node exists in tree
        - New parent is descendant of node: Would create cycle, must prevent
        - Root node relocation: Special handling for relocating root node
        """
        # Check if new_parent is a descendant of node (would create cycle)
        current = new_parent
        while current is not None:
            if current == node:
                return False  # Would create cycle
            current = current.parent
            
        # Remove from current parent
        if node.parent:
            node.parent.children.remove(node)
            
        # Add to new parent
        new_parent.children.append(node)
        node.parent = new_parent
        
        return True
    
    def balance_tree(self):
        """
        Balances the tree structure by redistributing nodes to maintain efficient access paths.
        Unlike basic reorganization, this focuses on overall tree efficiency and access patterns.
        
        Returns:
            dict: Information about the balancing operation
            
        Algorithm:
        1. Analyze current tree access patterns and depth distribution
        2. Identify frequently accessed deep nodes that should be moved higher
        3. Identify rarely accessed nodes that can be pushed deeper
        4. Reorganize tree to minimize average weighted access path length
        5. Return statistics about the reorganization
        
        Edge cases:
        - Conflicting access patterns: Need strategy when node has varied access patterns
        - Important relationship preservation: Maintain critical semantic relationships
        - Balance vs. semantic organization: Trade-off between efficiency and logical structure
        """
        # Implementation would go here
        balance_report = {
            'nodes_moved_up': 0,
            'nodes_moved_down': 0,
            'avg_depth_before': 0,
            'avg_depth_after': 0
        }
        
        # Calculate current average depth
        
        # Identify frequently accessed nodes that are too deep
        
        # Identify rarely accessed nodes that are too shallow
        
        # Perform node relocations
        
        # Calculate new average depth
        
        return balance_report