import re
import uuid
import json
from collections import deque
from datetime import datetime
from typing import List, Dict, Any, Optional
from supabase_client import supabase_client


class SupabaseTopicNode:
    """
    A TopicNode implementation that stores data in Supabase instead of memory.
    Maintains the same API as the original TopicNode but persists to database.
    """
    
    def __init__(self, topic_name: str, node_id: str = None, 
                 data: Dict = None, parent_id: str = None, metadata: Dict = None):
        """
        Initialize a new TopicNode.
        
        Args:
            topic_name (str): The name of this topic
            node_id (str, optional): UUID for this node. Generated if not provided.
            data (dict, optional): Data associated with this topic
            parent_id (str, optional): UUID of parent node
            metadata (dict, optional): Additional metadata about this topic
        """
        self.topic = topic_name
        self.node_id = node_id or str(uuid.uuid4())
        self.data = data or {}
        self.parent_id = parent_id
        self.metadata = metadata or {}
        self._children_cache = None
        self._parent_cache = None
        
    def save(self):
        """Save this node to Supabase"""
        try:
            data_to_save = {
                'id': self.node_id,
                'name': self.topic,
                'parent_id': self.parent_id,
                'description': self.data.get('description', ''),
                'memories': [self.data] if self.data else [],
                'metadata': self.metadata,
                'updated_at': datetime.now().isoformat()
            }
            
            # Try to update first, then insert if it doesn't exist
            result = supabase_client.client.table('memory_nodes').upsert(data_to_save).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"Error saving node {self.node_id}: {e}")
            return None
    
    @classmethod
    def load(cls, node_id: str):
        """Load a node from Supabase by ID"""
        try:
            result = supabase_client.client.table('memory_nodes')\
                .select('*')\
                .eq('id', node_id)\
                .execute()
            
            if result.data:
                node_data = result.data[0]
                memories = node_data.get('memories', [])
                data = memories[0] if memories else {}
                
                return cls(
                    topic_name=node_data['name'],
                    node_id=node_data['id'],
                    data=data,
                    parent_id=node_data.get('parent_id'),
                    metadata=node_data.get('metadata', {})
                )
            return None
        except Exception as e:
            print(f"Error loading node {node_id}: {e}")
            return None
    
    @classmethod
    def get_global_root(cls):
        """Get or create the global root node for shared memory"""
        try:
            # Try to find existing root node
            result = supabase_client.client.table('memory_nodes')\
                .select('*')\
                .is_('parent_id', None)\
                .execute()
            
            if result.data:
                node_data = result.data[0]
                memories = node_data.get('memories', [])
                data = memories[0] if memories else {}
                
                return cls(
                    topic_name=node_data['name'],
                    node_id=node_data['id'],
                    data=data,
                    parent_id=None,
                    metadata=node_data.get('metadata', {})
                )
            else:
                # Create new root node
                root = cls("Knowledge Base")
                root.save()
                return root
        except Exception as e:
            print(f"Error getting global root: {e}")
            # Return a new root node as fallback
            return cls("Knowledge Base")
    
    @property
    def children(self):
        """Get children nodes (cached)"""
        if self._children_cache is None:
            self._load_children()
        return self._children_cache
    
    def _load_children(self):
        """Load children from database"""
        try:
            result = supabase_client.client.table('memory_nodes')\
                .select('*')\
                .eq('parent_id', self.node_id)\
                .execute()
            
            self._children_cache = []
            for child_data in result.data:
                memories = child_data.get('memories', [])
                data = memories[0] if memories else {}
                
                child = SupabaseTopicNode(
                    topic_name=child_data['name'],
                    node_id=child_data['id'],
                    data=data,
                    parent_id=child_data.get('parent_id'),
                    metadata=child_data.get('metadata', {})
                )
                self._children_cache.append(child)
        except Exception as e:
            print(f"Error loading children for {self.node_id}: {e}")
            self._children_cache = []
    
    @property
    def parent(self):
        """Get parent node (cached)"""
        if self.parent_id is None:
            return None
        if self._parent_cache is None:
            self._parent_cache = SupabaseTopicNode.load(self.parent_id)
        return self._parent_cache
    
    def add_child(self, topic_name: str, data: Dict = None, metadata: Dict = None):
        """Add a child node"""
        child = SupabaseTopicNode(
            topic_name=topic_name,
            data=data,
            parent_id=self.node_id,
            metadata=metadata
        )
        child.save()
        
        # Invalidate children cache
        self._children_cache = None
        return child
    
    def find_child(self, topic_name: str):
        """Find a direct child by topic name"""
        for child in self.children:
            if child.topic == topic_name:
                return child
        return None
    
    def update_data(self, new_data: Dict):
        """Update the data associated with this topic"""
        self.data = new_data
        self.save()
    
    def get_path(self):
        """Get the full path from root to this node"""
        if self.parent is None:
            return [self.topic]
        else:
            return self.parent.get_path() + [self.topic]
    
    def get_depth(self):
        """Get the depth of this node in the tree"""
        if self.parent is None:
            return 0
        else:
            return 1 + self.parent.get_depth()
    
    def remove_child(self, topic_name: str):
        """Remove a child node by topic name"""
        child = self.find_child(topic_name)
        if child:
            try:
                # Delete from database
                supabase_client.client.table('memory_nodes')\
                    .delete()\
                    .eq('id', child.node_id)\
                    .execute()
                
                # Invalidate children cache
                self._children_cache = None
                return True
            except Exception as e:
                print(f"Error removing child {topic_name}: {e}")
        return False
    
    def get_all_descendants(self):
        """Get all descendants of this node"""
        descendants = []
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_all_descendants())
        return descendants
    
    def get_root(self):
        """Get the root node of the tree"""
        if self.parent is None:
            return self
        else:
            return self.parent.get_root()
    
    def find_topic(self, topic_name: str):
        """Find a topic in the subtree rooted at this node"""
        if self.topic == topic_name:
            return self
            
        # Check immediate children first
        for child in self.children:
            if child.topic == topic_name:
                return child
                
        # Then check descendants
        for child in self.children:
            result = child.find_topic(topic_name)
            if result:
                return result
                
        return None
    
    def print_tree(self, indent: int = 0):
        """Print the tree structure with indentation"""
        print(" " * indent + "- " + self.topic)
        for child in self.children:
            child.print_tree(indent + 2)
    
    def to_dict(self):
        """Convert this node and its subtree to a dictionary"""
        return {
            "topic": self.topic,
            "data": self.data,
            "metadata": self.metadata,
            "children": [child.to_dict() for child in self.children]
        }
    
    def get_relevant_context(
        self,
        query: str,
        depth: int = 2,
        max_results: int = 5,
        semantic_weight: float = 0.7,
        token_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Return up to ``max_results`` data objects drawn from this node and its
        descendants ≤ ``depth`` levels away that are most relevant to *query*.

        Enhanced Relevance Heuristic
        ----------------------------
        1. Use LLM to generate keywords and node categories
        2. Compute multi-dimensional scoring:
        a. Token overlap score
        b. Semantic similarity score
        c. Depth-based dampening
        d. Recency bonus

        Args:
            query (str): Search query
            depth (int): Maximum tree depth to search
            max_results (int): Maximum number of results to return
            semantic_weight (float): Weight for semantic similarity (0-1)
            token_weight (float): Weight for token overlap (0-1)

        Returns
        -------
        List of most relevant data objects
        """
        print(f"🔍 Starting enhanced context search for query: '{query}'")
        
        # Step 1: Use LLM to predict likely node categories and keywords
        likely_keywords = self._get_likely_keywords_from_llm(query)
        if not likely_keywords:
            print("❌ No keywords generated by LLM, falling back to query tokens")
            likely_keywords = set(re.findall(r'\w+', query.lower()))
        
        print(f"🎯 LLM-generated keywords: {likely_keywords}")
        
        # ---- 2. Semantic encoding (placeholder) -----
        def semantic_similarity(text1: str, text2: str) -> float:
            """
            Compute semantic similarity between two texts.
            Placeholder method that could be replaced with more advanced embedding techniques.
            """
            # Simple jaccard similarity as placeholder
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            return len(words1.intersection(words2)) / len(words1.union(words2))

        # ---- 3. Breadth-first traversal --------------------------------------
        queue = deque([(self, 0)])
        scored_items = []  # (total_score, -depth, timestamp_numeric, node.data)
        visited = set()

        while queue:
            node, d = queue.popleft()
            
            # Skip if already visited or too deep
            if node.node_id in visited or d > depth:
                continue
            
            visited.add(node.node_id)
            print(f"Exploring '{node.topic}' at depth {d}")

            # Assemble searchable text for this node
            data_text = "".join(map(str, node.data.values())) if isinstance(node.data, dict) else str(node.data)
            full_text = f"{node.topic} {data_text}".lower()

            # Token overlap score
            tokens = set(re.findall(r'\w+', query.lower()))
            token_score = sum(1 for t in tokens if t in full_text) / len(tokens) if tokens else 0

            # Semantic similarity score
            semantic_score = semantic_similarity(query, full_text)

            # Combine scores with weighted average
            combined_score = (
                semantic_weight * semantic_score + 
                token_weight * token_score
            )

            # Depth dampening (nodes deeper in tree get lower score)
            depth_factor = 1 / (1 + 0.2 * d)  # Mild dampening

            # Recency bonus (use last_updated from metadata if available)
            ts = node.metadata.get('last_updated', '')
            
            # Convert timestamp to numeric value for safe comparison
            try:
                timestamp_numeric = float(ts) if ts else 0.0
            except (ValueError, TypeError):
                timestamp_numeric = 0.0
            
            timestamp_score = 1 if timestamp_numeric == 0 else 1 / (1 + timestamp_numeric)

            # Keyword matching bonus
            keyword_bonus = 1.5 if any(kw.lower() in node.topic.lower() for kw in likely_keywords) else 1

            total_score = combined_score * depth_factor * timestamp_score * keyword_bonus

            if total_score > 0:
                print(f"Added '{node.topic}' to results (score: {total_score})")
                # Use timestamp_numeric instead of ts for safe sorting
                scored_items.append((total_score, -d, timestamp_numeric, node.data))

            # Enqueue children that match keywords or are not too deep
            for child in node.children:
                if child.node_id not in visited:
                    # Explore if child matches keywords or is not too deep
                    if any(kw.lower() in child.topic.lower() for kw in likely_keywords) or d < depth:
                        queue.append((child, d + 1))

        # Sort and limit results - this should now work without comparison errors
        try:
            scored_items.sort(reverse=True)
        except TypeError as e:
            print(f"⚠️ Sorting error: {e}, using score-only sorting")
            # Fallback: sort only by score
            scored_items.sort(key=lambda x: x[0], reverse=True)
        
        results = [item[3] for item in scored_items[:max_results]]
        
        print(f"✅ Returning {len(results)} context items")
        return results

    def _get_likely_keywords_from_llm(self, query: str) -> set:
        """Use Mistral AI to generate likely node categories and keywords for the search"""
        try:
            import os
            from mistralai import Mistral
            
            # Get API key
            api_key = os.environ.get('MISTRAL_API_KEY')
            if not api_key:
                print("⚠️  MISTRAL_API_KEY not found, skipping LLM keyword generation")
                return set()
            
            # Initialize Mistral client
            client = Mistral(api_key=api_key)
            
            # Create prompt for keyword generation
            print("🤖 Calling Mistral AI for keyword generation...")
            
            # Make API call with placeholder agent ID
            agent_key = "ag:a2eb8171:20250611:hozie-generate-possible-topic-nodes-for-query:8b9ba8bc"
            response = client.agents.complete(
                agent_id=agent_key, 
                messages=[
                    {
                        "role": "user",
                        "content": query
                    }
                ]
            )
            output = response.choices[0].message.content
            
            print(f"[Brain] raw path generation output: {output}")
            
            # Extract JSON array from the response
            match = re.search(r'\[.*\]', output, re.DOTALL)
            if match:
                json_str = match.group(0)
                try:
                    keywords = json.loads(json_str)
                    #If its a valid JSON array, return the keywords + the root node
                    if isinstance(keywords, list) and all(isinstance(item, str) for item in keywords) and len(keywords) >= 2:
                        keywords.append('Knowledge Base')
                        return keywords
                except json.JSONDecodeError:
                    print(f"❌ Error parsing JSON from LLM response: {output}")
                    pass            
        except Exception as e:
            print(f"❌ Error calling Mistral AI: {e}")
        
        return set()
    
    def _targeted_bfs_search(self, query: str, keywords: set, depth: int, max_results: int) -> List[Dict[str, Any]]:
        """Perform simple breadth-first search guided by keywords"""
        print(f"🔎 Starting simple BFS with keywords: {keywords}")
        
        queue = deque([(self, 0)])  # (node, depth)
        results = []
        visited = set()
        
        while queue and len(results) < max_results:
            node, current_depth = queue.popleft()
            
            # Skip if already visited or too deep
            if node.node_id in visited or current_depth > depth:
                continue
                
            visited.add(node.node_id)
            print(f"Exploring '{node.topic}' at depth {current_depth}")
            
            # Add to results if node has data
            if node.data:
                results.append(node.data)
                print(f"Added '{node.topic}' to results")
            
            # Decide whether to explore children
            if self._should_explore_simple(node, keywords, current_depth):
                print(f"Exploring {len(node.children)} children")
                for child in node.children:
                    if child.node_id not in visited:
                        queue.append((child, current_depth + 1))
            else:
                print(f"  ⏭️  Skipping children")
        
        print(f"✅ Returning {len(results)} context items")
        return results
    
    def _should_explore_simple(self, node, keywords, current_depth):
        """Simple rule: only explore if node topic is in keywords or depth > 3"""
        # Always explore if depth > 3 (deep exploration)
        if current_depth > 3:
            return True
        
        # Check if node topic contains any keyword
        topic = node.topic
        for keyword in keywords:
            if topic in keyword or keyword in topic:
                print(f"    🎯 Topic '{node.topic}' matches keyword '{keyword}'")
                return True
        
        return False
    
    def delete(self):
        """Delete this node and all its descendants from the database"""
        try:
            # First delete all descendants
            for child in self.children:
                child.delete()
            
            # Then delete this node
            supabase_client.client.table('memory_nodes')\
                .delete()\
                .eq('id', self.node_id)\
                .execute()
            return True
        except Exception as e:
            print(f"Error deleting node {self.node_id}: {e}")
            return False
    