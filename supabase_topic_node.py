import uuid
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
        
    def save(self) -> Optional[Dict[str, Any]]:
        """
        Save this node to Supabase
        
        Returns:
            Optional[Dict[str, Any]]: The saved node data if successful, None otherwise"""
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
            
            result = supabase_client.client.table('memory_nodes').upsert(data_to_save).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"Error saving node {self.node_id}: {e}")
            return None
    
    @classmethod
    def load(cls, node_id: str) -> Optional['SupabaseTopicNode']:
        """
        Load a node from Supabase by ID

        Args:
            node_id (str): The UUID of the node to load

        Returns:
            Optional[SupabaseTopicNode]: The loaded node, or None if not found
        """
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
    def get_global_root(cls) -> 'SupabaseTopicNode':
        """
        Get or create the global root node for shared memory
        
        Returns:
            SupabaseTopicNode: The global root node
        """
        try:
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
                root = cls("Knowledge Base")
                root.save()
                return root
        except Exception as e:
            print(f"Error getting global root: {e}")
            return cls("Knowledge Base")
    
    @property
    def children(self) -> List['SupabaseTopicNode']:
        """
        Get children nodes (cached)
        
        Returns:
            List[SupabaseTopicNode]: List of child nodes
        """
        if self._children_cache is None:
            self._load_children()
        return self._children_cache
    
    def _load_children(self) -> None:
        """
        Load children from database
        """
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
    def parent(self) -> Optional['SupabaseTopicNode']:
        """
        Get parent node (cached)
        
        Returns:
            Optional[SupabaseTopicNode]: The parent node, or None if no parent
        """
        if self.parent_id is None:
            return None
        if self._parent_cache is None:
            self._parent_cache = SupabaseTopicNode.load(self.parent_id)
        return self._parent_cache
    
    def add_child(self, topic_name: str, data: Dict = None, metadata: Dict = None) -> 'SupabaseTopicNode':
        """
        Add a child node
        
        Args:
            topic_name (str): The name of the child topic
            data (dict, optional): Data associated with the child topic
            metadata (dict, optional): Additional metadata for the child topic

        Returns:
            SupabaseTopicNode: The newly created child node
        """
        child = SupabaseTopicNode(
            topic_name=topic_name,
            data=data,
            parent_id=self.node_id,
            metadata=metadata
        )
        child.save()
        
        self._children_cache = None
        return child
    
    def find_child(self, topic_name: str) -> Optional['SupabaseTopicNode']:
        """
        Find a direct child by topic name

        Args:
            topic_name (str): The name of the child topic to find

        Returns:
            Optional[SupabaseTopicNode]: The found child node, or None if not found
        """
        for child in self.children:
            if child.topic == topic_name:
                return child
        return None
    
    def update_data(self, new_data: Dict) -> None:
        """
        Update the data associated with this topic
        
        Args:
            new_data (dict): New data to associate with this topic
        """
        self.data = new_data
        self.save()
    
    def get_path(self) -> List[str]:
        """
        Get the full path from root to this node
        
        Returns:
            List[str]: List of topic names from root to this node
        """
        if self.parent is None:
            return [self.topic]
        else:
            return self.parent.get_path() + [self.topic]
    
    def get_depth(self) -> int:
        """
        Get the depth of this node in the tree

        Returns:
            int: The depth of the node
        """
        if self.parent is None:
            return 0
        else:
            return 1 + self.parent.get_depth()
    
    def remove_child(self, topic_name: str) -> bool:
        """
        Remove a child node by topic name
        
        Args:
            topic_name (str): The name of the child topic to remove

        Returns:
            bool: True if the child was removed, False otherwise
        """
        child = self.find_child(topic_name)
        if child:
            try:
                supabase_client.client.table('memory_nodes')\
                    .delete()\
                    .eq('id', child.node_id)\
                    .execute()
                
                self._children_cache = None
                return True
            except Exception as e:
                print(f"Error removing child {topic_name}: {e}")
        return False
    
    def get_all_descendants(self) -> List['SupabaseTopicNode']:
        """
        Get all descendants of this node
        
        Returns:
            List[SupabaseTopicNode]: List of all descendant nodes
        """
        descendants = []
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_all_descendants())
        return descendants
    
    def get_root(self) -> 'SupabaseTopicNode':
        """
        Get the root node of the tree
        
        Returns:
            SupabaseTopicNode: The root node of the tree
        """
        if self.parent is None:
            return self
        else:
            return self.parent.get_root()
    
    def find_topic(self, topic_name: str) -> Optional['SupabaseTopicNode']:
        """
        Find a topic in the subtree rooted at this node
        
        Args:
            topic_name (str): The name of the topic to find

        Returns:
            Optional[SupabaseTopicNode]: The found node, or None if not found
        """
        if self.topic == topic_name:
            return self
            
        for child in self.children:
            if child.topic == topic_name:
                return child
                
        for child in self.children:
            result = child.find_topic(topic_name)
            if result:
                return result
                
        return None
    
    def print_tree(self, indent: int = 0):
        """
        Print the tree structure with indentation
        
        Args:
            indent (int): Current indentation level
        """
        print(" " * indent + "- " + self.topic)
        for child in self.children:
            child.print_tree(indent + 2)
    
    def to_dict(self):
        """
        Convert this node and its subtree to a dictionary
        
        Returns:
            Dict: Dictionary representation of this node and its children
        """
        return {
            "topic": self.topic,
            "data": self.data,
            "metadata": self.metadata,
            "children": [child.to_dict() for child in self.children]
        }
      
    def delete(self) -> bool:
        """
        Delete this node and all its descendants from the database
        
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            for child in self.children:
                child.delete()
            
            supabase_client.client.table('memory_nodes')\
                .delete()\
                .eq('id', self.node_id)\
                .execute()
            return True
        except Exception as e:
            print(f"Error deleting node {self.node_id}: {e}")
            return False