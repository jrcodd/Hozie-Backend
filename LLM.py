#Using ministral causes a lot of errors when trying to get a specific json output even if it is a little faster. Useing Nemo because it can do it better.
import json
import os
import re
import textwrap
import time
import random
import asyncio
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence, Tuple, Optional

import requests
from bs4 import BeautifulSoup
from mistralai import Mistral, UserMessage

from supabase_topic_node import SupabaseTopicNode 
import concurrent.futures


def _get_time() -> str:
    return datetime.now(timezone.utc).isoformat()

class Brain:
    """
    Hozie's Brain

    1.  Try to answer from memory (1 API call to Mistral (quick bc of short propt)to get likely nodes then modified bfs the tree skipping nodes that are not relevant, another (longer) API call to Mistral to generate the answer with this context)
    IF no answer found in memory:
    2.  Query rewriting (user → search query) (API call to Mistral)
    3.  Web search + scraping (DuckDuckGo + BeautifulSoup)
    4.  Summarisation → JSON suitable for SupabaseTopicNode storage (API call to Mistral)
    5.  Retrieval-Augmented Generation to answer the user (step 1)

    entry point is Brain.answer(question: str) → str
    """
    debug = True
    def __init__(self, debug: bool = True) -> None:
        """
        Connect to Mistral API and create or load memory tree.
        
        Args:
            api_key: Mistral API key. If not provided, it will be read from the environment variable MISTRAL_API_KEY.
        """
        self.api_key = os.environ.get('MISTRAL_API_KEY')
        self._init_mistral()
        self.debug = debug
        if(debug):
            print(f"[Brain] using Supabase storage for global memory")
        self.memory = SupabaseTopicNode.get_global_root()
        self.chat_history = None
        # Thread lock to prevent concurrent memory insertions that could create duplicates
        self._memory_lock = threading.Lock()
    
    def _init_mistral(self) -> None:
        """
        Connect to Mistral API
        """
        if not self.api_key:
            raise ValueError("Mistral API key is required. Set MISTRAL_API_KEY environment variable")

        self.client = Mistral(api_key=self.api_key)
        if(self.debug):
            print("[Brain] Mistral API connected.✓")

    def _rewrite_query(self, user_query: str) -> str:
        """
        Uses Mistral API to generate a search query based on the user's question. 

        TODO: implement a list of search queries to improve coverage for longer/cmopound questions. Also implement different search phrases like doctype:pdf or scholarly articles or stuff like that for research based questions.

        0.6 temp mistral nemo with this system prompt:
           You are an expert search assistant. Convert the user question into a short, keyword-focused web search query that will fully answer their question. Respond with ONLY the query. No quotations

        Args:
            user_query: The original question asked by the user.
            
        Returns:
            A rewritten search query that is more suitable for web searching
        """

        prompt = f"{user_query}"
        agent_key = "ag:a2eb8171:20250526:hozie-generate-search-qwery:f121b601"
        chat_response = self.client.agents.complete(
            agent_id=agent_key,
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt}",
                },
            ],
        )
        query = chat_response.choices[0].message.content
        return query

    def _search(self, query: str, max_results: int = 6) -> List[Dict]:
        """
        Return a list of websites using brave search API.

        Args:
            query: The search query to use for finding relevant websites.
            max_results: The maximum number of search results to return (default is 6).

        Returns:
            List of dictionaries containing URLs and titles of the search results.
        """

        api_key = os.environ.get('BRAVE_SEARCH_API_KEY')
    
        headers = {
            'X-Subscription-Token': api_key,
            'Accept': 'application/json'
        }
        
        params = {
            'q': query,
            'count': max_results,
            'safesearch': 'moderate',
            'search_lang': 'en',
            'country': 'US'
        }
        
        try:
            if(self.debug):
                print(f"[Brain] Searching Brave for: {query}")
            response = requests.get(
                'https://api.search.brave.com/res/v1/web/search',
                headers=headers,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                web_results = data.get('web', {}).get('results', [])
                if(self.debug):
                    print(f"[Brain] Brave returned {len(web_results)} results")
                
                for result in web_results[:max_results]:
                    title = result.get('title', '')
                    url = result.get('url', '')
                    description = result.get('description', '')
                    
                    if url and title:
                        results.append({
                            'url': url,
                            'title': title,
                            'snippet': description
                        })
                        if(self.debug): 
                            print(f"[Brain] Added result: {title[:50]}...")

                return results
            else:
                if(self.debug): print(f"[Brain] Brave API error: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            if(self.debug): print(f"[Brain] Brave search error: {e}")
            return []
    
    def _scrape(self, url: str, char_limit: int = 12000) -> str:
        """
        Scrape a webpage and extract the main content text. Char limit should be low to keep speed of the llm, but also should be high to get all the content from longer websites like wikipedia articles. 12000 seems to work good from what I have seen.

        Args:
            url: The URL of the webpage to scrape.
            char_limit: The maximum number of characters to return from the scraped content (default is 12000).

        Returns:
            The main text content extracted from the webpage, limited to char_limit characters.
        """

        #User agents to reduce the chance of being blocked by websites (works ok but I might look into using puppeteer for getting blocked less)
        USER_AGENTS = [
            # Chrome
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            # Firefox
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
            # Safari
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
            # Edge
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/123.0.0.0",
        ]
        
        if(self.debug): print(f"[Brain] scraping: {url}")
        
        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "DNT": "1",
        }
        
        def fetch_url(retry=False) -> Optional[str]:
            """
            Fetch the URL with retry logic for rate limits or access issues.

            Args:
                retry: If true, this is a retried attempt so switch the user agent

            Returns:
                The HTML content of the page, or None if an error occurred.
            """
            if retry:
                current_ua = headers["User-Agent"]
                alternative_user_agents = []
                for ua_option in USER_AGENTS:
                    if ua_option != current_ua:
                        alternative_user_agents.append(ua_option)

                new_user_agent = random.choice(alternative_user_agents)
                headers["User-Agent"] = new_user_agent
                if(self.debug): print(f"[Brain] retrying with different user agent")   

            try:
                response = requests.get(url, timeout=10, headers=headers, allow_redirects=True)
                response.raise_for_status()
                
                content_type = response.headers.get('Content-Type', '').lower()
                if 'text/html' not in content_type:
                    if(self.debug): print(f"[Brain] skipping non-HTML content: {content_type}")
                    return None
                    
                return response.text
            except requests.exceptions.HTTPError as e:
                #Only retry once to prevent an infinite loop
                if not retry and hasattr(e, 'response') and e.response.status_code in [403, 429]:
                    return fetch_url(retry=True)
                if(self.debug): print(f"[Brain] HTTP error: {e}")
                return None
            except Exception as e:
                if(self.debug): print(f"[Brain] request error: {e}")
                return None
        
        html_content = fetch_url()
        if not html_content:
            return ""
                    
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            
            for selector in ['script', 'style', 'noscript', 'iframe', 'nav', 'header', 'footer', 'aside']:
                for tag in soup.select(selector):
                    tag.decompose()
            
            text = ""
            # try to find main content
            main_selectors = ['article', 'main', '[role="main"]', '.content', '#content', '.post-content']
            for selector in main_selectors:
                main_content = soup.select_one(selector)
                if main_content and len(' '.join(main_content.stripped_strings)) > 200:
                    text = ' '.join(main_content.stripped_strings)
                    break
            
            # if no main content found, try to extract text from paragraphs
            if not text:
                paragraphs = []
                for tag in soup.select('p, h1, h2, h3, h4, h5, h6, li'):
                    p_text = tag.get_text(strip=True)
                    if p_text and len(p_text) > 20:  # Skip very short fragments
                        paragraphs.append(p_text)
                
                if paragraphs:
                    text = ' '.join(paragraphs)
            
            # fallback on body text if nothing was found
            if not text and soup.body:
                text = ' '.join(soup.body.stripped_strings)
            
            #clean the text now
            if text:
                # Replace multiple whitespace with a single space
                text = re.sub(r'\s+', ' ', text)
                # Remove space before punctuation
                text = re.sub(r'\s([.,;:!?])', r'\1', text)
                #remove ad content
                text = re.sub(r'\b(ad|sponsored|advertisement|promo|promotion)\b', '', text, flags=re.IGNORECASE)
                # Return limited text
                return text[:char_limit]
                
            # Last resort fallback: strip HTML tags with regex
            if not text and html_content:
                text = re.sub(r'<[^>]*>', ' ', html_content)
                text = re.sub(r'\s+', ' ', text).strip()
                return text[:char_limit]
                
            return ""
            
        except Exception as e:
            if(self.debug): print(f"[Brain] error parsing content: {e}")
            return ""

    def _summarise(self, raw_text: str) -> Dict:
        """
        Extract a JSON-like summary suitable for SupabaseTopicNode.data.
        
        Args:
            raw_text: The raw text content to summarize.

        Returns:
            A dictionary containing the summary with keys: main_idea, description, bullet_points, source.
        """


        if len(raw_text) < 200: 
            if(self.debug): print(f"[Brain] text too short: ({len(raw_text)} chars)")
            return {}

        if(self.debug): print(f"[Brain] attempting to summarize {len(raw_text)} chars of text")
        
        prompt = f"{raw_text}"
        output = ""
        """
        mistral small is cheaper for long inputs
        0.3 temp mistral small prompt:
        Your job is to extract structured knowledge from the text that was scraped from the web. Respond ONLY with a single valid JSON object with these exact keys:
            "main_idea": (the main idea of the text),
            "description": (keep it short, but include enough information to understand the topic and be able to answer questions about it),
            "bullet_points": (Array of main points from the text that are relevant to the topic and would be useful to use to answer questions.)
            "source": (source of the information if available, otherwise empty string)
            
            Example output:
            {
                "main_idea": "Theory of Relativity",
                "description": "The theory of relativity is a scientific theory of gravitation that was developed by Albert Einstein in the early 20th century.",
                "bullet_points": [
                    "It consists of two theories: special relativity and general relativity.",
                    "Special relativity deals with the physics of objects moving at constant speeds, particularly at speeds close to the speed of light.",
                    "General relativity extends the principles of special relativity to include acceleration and gravity."
                ],
                "source": "https://en.wikipedia.org/wiki/Theory_of_relativity"
            }

            Make sure your response is ONLY valid JSON that can be parsed by json.loads(). Do not include any text before or after the JSON. Make sure all brackets are closed and that the JSON is valid.

        """
        try:
            if(self.debug): print("[Brain] generating summary with LLM...")
            agent_key = "ag:a2eb8171:20250526:hozie-web-summary:d913c3b8"
            chat_response = self.client.agents.complete(
                agent_id=agent_key,
                messages=[
                    {
                        "role": "user",
                        "content": f"{prompt}",
                    },
                ],
            )
            response = chat_response.choices[0].message.content
            if(self.debug): print(f"[Brain] summarized output length: {len(response)} chars")
            
            try:
                data = json.loads(response)
                if(self.debug): print(f"[Brain] successfully extracted JSON with keys: {list(data.keys())}")
                
                required_keys = ["main_idea", "description", "bullet_points", "source"]
                for key in required_keys:
                    if key not in data:
                        data[key] = "" if key != "bullet_points" else []
                
                if not isinstance(data["bullet_points"], list):
                    if(self.debug): print("[Brain] bullet_points is not formatted correctly, trying to convert to list")
                    data["bullet_points"] = [data["bullet_points"]] if data["bullet_points"] else []
            
                return data
                
            except json.JSONDecodeError as e:
                if(self.debug): print(f"[Brain] JSON parsing error: {e}")

        except Exception as e:
            if(self.debug): print(f"[Brain] Error during summarization: {e}")
            if not output:
                if(self.debug): print("[Brain] No output generated, returning empty node")
                return {
                    "main_idea": "None",
                    "description": f"Failed to process content: {str(e)}",
                    "bullet_points": [],
                    "source": "error"
                }
            
    def _check_similar_path(self, node: SupabaseTopicNode, target_path: Sequence[str], current_path: List[str] = None) -> Tuple[bool, SupabaseTopicNode, List[str]]:
        """
        Check if a similar path already exists in the tree.

        Args:
            node: The current node in the tree to check.
            target_path: The path we are trying to match against the tree.
            current_path: The path we have traversed so far (used for recursion).
        
        Returns:
            Tuple[bool, SupabaseTopicNode, List[str]]: 
                - True if a similar path was found
                - The closest matching node
                - The path to that node
        """
        if current_path is None:
            current_path = [node.topic]
        
        # If we've reached the end of the target path, we've found a match
        if not target_path:
            return True, node, current_path
        
        # Try to find children that match or are similar to the next part of the path
        next_topic = target_path[0]
        
        # Direct match
        exact_child = node.find_child(next_topic)
        if exact_child:
            return self._check_similar_path(exact_child, target_path[1:], current_path + [exact_child.topic])
        
        # Look for similar topics by name (case insensitive)
        for child in node.children:
            # 1. Check for exact match ignoring case
            if child.topic.lower() == next_topic.lower():
                if(self.debug): print(f"[Brain] found case-insensitive match: {child.topic} for {next_topic}")
                return self._check_similar_path(child, target_path[1:], current_path + [child.topic])
                
            # 2. Check if one is a prefix/suffix of the other
            # This will match "Black Holes" with "Black Hole" or "Black Holes Formation"
            if child.topic.lower().startswith(next_topic.lower()) or next_topic.lower().startswith(child.topic.lower()):
                if(self.debug): print(f"[Brain] found prefix/suffix match: {child.topic} for {next_topic}")
                return self._check_similar_path(child, target_path[1:], current_path + [child.topic])
                
            # 3. Check for word-level similarity
            child_words = set(re.findall(r'\w+', child.topic.lower()))
            target_words = set(re.findall(r'\w+', next_topic.lower()))
            
            # If there's significant overlap, consider it similar
            if child_words and target_words:
                overlap = child_words.intersection(target_words)
                similarity = len(overlap) / max(len(child_words), len(target_words))
                
                if similarity > 0.8:  
                    if(self.debug): print(f"[Brain] found similar node: {child.topic} for {next_topic} (similarity: {similarity:.2f})")
                    return self._check_similar_path(child, target_path[1:], current_path + [child.topic])
        
        # 4. Special case handling for common hierarchical relationships
        science_categories = ["physics", "astronomy", "chemistry", "biology", "climetology", "geology", "earth science", "environmental science"]
        if (node.topic.lower() == "science" and next_topic.lower() in science_categories) or \
           (next_topic.lower() == "science" and node.topic.lower() in science_categories):
            # If this is a science-related path, check if we're duplicating the hierarchy
            if(self.debug): print(f"[Brain] detected potential science category duplication: {node.topic} vs {next_topic}")
            
            # For "Science/Physics" and "Physics" patterns, prefer the more specific
            if node.topic.lower() == "science" and next_topic.lower() in science_categories:
                for child in node.children:
                    if child.topic.lower() == next_topic.lower():
                        if(self.debug): print(f"[Brain] found science subcategory: {child.topic}")
                        return self._check_similar_path(child, target_path[1:], current_path + [child.topic])
            
        # No similar path found
        return False, node, current_path
    
    def _find_similar_child(self, parent_node: SupabaseTopicNode, target_topic: str) -> Optional[SupabaseTopicNode]:
        """
        Find a similar child node to avoid creating duplicates.
        
        Args:
            parent_node: The parent node whose children to search
            target_topic: The topic name we're looking for a similar match to
            
        Returns:
            Optional[SupabaseTopicNode]: The similar child node if found, None otherwise
        """
        for child in parent_node.children:
            # 1. Check for exact match ignoring case
            if child.topic.lower() == target_topic.lower():
                if(self.debug): print(f"[Brain] found case-insensitive match: {child.topic} for {target_topic}")
                return child
                
            # 2. Check if one is a prefix/suffix of the other
            if child.topic.lower().startswith(target_topic.lower()) or target_topic.lower().startswith(child.topic.lower()):
                if(self.debug): print(f"[Brain] found prefix/suffix match: {child.topic} for {target_topic}")
                return child
                
            # 3. Check for word-level similarity
            child_words = set(re.findall(r'\w+', child.topic.lower()))
            target_words = set(re.findall(r'\w+', target_topic.lower()))
            
            if child_words and target_words:
                overlap = child_words.intersection(target_words)
                similarity = len(overlap) / max(len(child_words), len(target_words))
                
                if similarity > 0.8:  
                    if(self.debug): print(f"[Brain] found similar node: {child.topic} for {target_topic} (similarity: {similarity:.2f})")
                    return child
        
        return None
        
    def _insert_memory(self, topic_path: Sequence[str], data: Dict, source_url: str, confidence: float = 0.80) -> None:
        """
        Create/merge SupabaseTopicNodes along topic_path and attach data.
        Uses thread lock to prevent concurrent insertions from creating duplicate paths.
        
        Args:
            topic_path: List of topics representing the path in the memory tree.
            data: Dictionary containing the data to store at the leaf node.
            source_url: URL of the source where the data was retrieved from.
            confidence: Confidence level of the data (default is 0.80).
        """
        with self._memory_lock:  # Prevent concurrent memory insertions
            if not topic_path:
                if(self.debug): 
                    print("[Brain] ERROR: Empty topic path provided, using fallback path")
                topic_path = ["Uncategorized", "WebContent"]
            
            # Ensure topic_path is a list of strings
            if not isinstance(topic_path, (list, tuple)):
                if(self.debug):
                    print(f"[Brain] ERROR: topic_path is not iterable: {type(topic_path)}, converting to list")
                topic_path = [str(topic_path)]
            elif not all(isinstance(item, str) for item in topic_path):
                if(self.debug):
                    print("[Brain] ERROR: topic_path contains non-string items, converting")
                topic_path = [str(item) for item in topic_path]
            
            if not data:
                if(self.debug):
                    print("[Brain] ERROR: Empty data provided, skipping memory insertion")
                return
            
            try:
                # Skip the root if it matches
                path_to_create = topic_path[1:] if topic_path[0] == self.memory.topic else topic_path
                
                # Build path step by step, checking for existing nodes at each level
                current_node = self.memory
                
                for i, topic in enumerate(path_to_create):
                    # Refresh children cache to get latest state from database
                    current_node._children_cache = None
                    
                    # Check if child already exists (exact match first)
                    existing_child = current_node.find_child(topic)
                    
                    if existing_child:
                        current_node = existing_child
                    else:
                        # Check for similar existing children to avoid duplicates
                        similar_child = self._find_similar_child(current_node, topic)
                        
                        if similar_child:
                            if(self.debug): 
                                print(f"[Brain] found similar existing node: {similar_child.topic} (for {topic})")
                            current_node = similar_child
                        else:
                            # Only add data to the leaf node (last topic in path)
                            if i == len(path_to_create) - 1:
                                current_node = current_node.add_child(topic, data)
                            else:
                                current_node = current_node.add_child(topic)
                
                target_node = current_node

                # Merge data into the target node
                if isinstance(target_node.data, dict) and target_node.data:
                    if(self.debug): 
                        print(f"[Brain] merging with existing data ({len(target_node.data)} keys)")
                    target_node.data.update(data)
                else:
                    target_node.data = data
                
                # Add metadata
                meta_entry = {
                    "url": source_url,
                    "timestamp": _get_time(),
                    "confidence": confidence,
                }
                target_node.metadata.setdefault("sources", []).append(meta_entry)
                target_node.metadata["last_updated"] = _get_time()
                
                # Save the updated node
                target_node.save()
                    
            except Exception as e:
                if(self.debug): print(f"[Brain] ERROR during memory insertion: {e}")

    def _generate_topic_path(self, query: str, title: str, data: Dict) -> List[str]:
        """
        Generate a structured topic path based on content analysis.
        This method uses Mistral to generate a topic path based on the user's query and the content's title and summary.
        
        Args:
            query: The user's search query or question.
            title: The title of the content being analyzed.
            data: A dictionary containing the content's description and bullet points.
        
        Returns:
            A list of strings representing the structured topic path.
        """
        prompt = textwrap.dedent(
            f"""
            USER QUERY: {query}
            CONTENT TITLE: {title}
            CONTENT SUMMARY: {data.get('description', '')}
            CONTENT BULLET POINTS: {', '.join(data.get('bullet_points', [])[:3])}
            """
        )
        if(self.debug): print(f"data for content generation: {prompt}")
        prompt += textwrap.dedent(
        """
        Create a logical knowledge hierarchy that for the given query and content:
            1. Starts with one of these broad domains: Science, Humanities, Technology, Arts, Social Sciences, Food, Health, Business, Education, Politics, etc.
            2. Follows with a subdomain (e.g., Physics, Ancient History, Computer Science, Visual Arts, Sociology)
            3. Continues with a topic area (e.g., Quantum Mechanics, Roman Empire, Machine Learning, Painting Techniques, Social Behavior)
            4. Ends with the specific concept (e.g., Wave Function, Julius Caesar, Neural Networks, Impressionism, Group Dynamics)
        The path should be a single JSON array of strings from general to specific, such as: ["Science", "Physics", "Quantum Mechanics", "Wave Function"]

        Analyze the following information and generate a hierarchical knowledge path for storing in a knowledge tree. The path should follow academic taxonomy conventions starting with broad domains and narrowing to specifics Keep it short.
        Return ONLY a single JSON array of path strings from general to specific, such as: ["Science", "Astronomy", "Celestial Objects", "Black Holes"]
        """
        )

        json_schema={
            "type": "array",
            "items": {"type": "string"},
        }
        output = self.sync_llm_call(prompt, temp=0.15, output_type="json_object", json_schema=json_schema)
        if(self.debug): 
            print(f"[Brain] topic path generation output: {output}")
        try:
            path = json.loads(output.strip())
            if isinstance(path, list) and all(isinstance(item, str) for item in path) and len(path) >= 2:
                if(self.debug): print(f"[Brain] generated structured path: {path}")
                return path
        except json.JSONDecodeError:
            if(self.debug): print(f"[Brain] Failed to parse JSON from topic path generation: {output}")
            pass
        
        # Fallback: create a simple path based on the query
        if(self.debug): print("[Brain] Using fallback topic path generation")
        fallback_path = ["General", "Web Content"]
        if query:
            # Try to extract a topic from the query
            query_words = query.replace("Tell me about", "").replace("What is", "").replace("How", "").strip()
            if query_words:
                fallback_path = ["General", query_words.split()[0].title()]
        
        return fallback_path

    def _retrieve_context(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve relevant context from memory tree.
        This method searches the memory tree for relevant nodes based on the user's query using a modified breadth-first search approach.

        Args:
            query: The user's search query or question.
            k: The number of context chunks to return (default is 5).
        
        Returns:
            A list of dictionaries containing relevant context chunks from the memory tree.
        """
        if(self.debug): print(f"[Brain] retrieving context for query: '{query}'")
        try: 
            contexts = self.get_relevant_context(query, max_results=k*2)

            if(self.debug):
                print(f"[Brain] found {len(contexts)} context chunks")
                for i in contexts:
                    print(f"[Brain] Context Chunk Title: {i.topic}")
                    desc = i.data.get('description', ''),
                    print(f"[Brain] Context Chunk: {desc}")

            if contexts:
                unique_contexts = []
                description_set = set()   

                for ctx in contexts:
                    if ctx.data.get('description') and ctx.data.get('bullet_points'):
                        desc = ctx.data.get('description', '')
                        if not desc or len(desc) <= 30:
                            unique_contexts.append(ctx)
                            continue

                        fingerprint = desc[:40].strip().lower()
                        if fingerprint in description_set:
                            if(self.debug): print(f"[Brain] skipping duplicate context: {fingerprint}")
                            continue
                        
                        description_set.add(fingerprint)
                        unique_contexts.append(ctx)
                                         
                results = unique_contexts[:k]
                if(self.debug): print(f"[Brain] returning top {len(results)} unique context chunks")
                
                return results
            else:
                if(self.debug): print("[Brain] no relevant context found")
                #dont call web search here bc it could cause an infinite loop if something is broken
                return []
            
        except Exception as e:
            import traceback
            if(self.debug): print(f"[Brain] error during context retrieval: {e}")
            if(self.debug): print(f"Stack trace:\n{traceback.format_exc()}")

    def answer(self, user_question: str, message_history: Dict[str, str], max_search_results: int = 4, stream: bool = False) -> str:
            """
            Main entry-point: answer a user question using RAG.
            Optimized to return responses faster by deferring knowledge storage to background.
            
            Args:
                user_question (str): The user's question to answer
                max_search_results (int, optional): Maximum number of websites to search when looking for information. Defaults to 5.
            Returns:
                str: The generated answer to the user's question.
            """
            total_start_time = datetime.now()
            step_times = {}  # Track individual step durations
            
            if(self.debug): print(f"[Brain] Starting answer process at {total_start_time.strftime('%H:%M:%S.%f')[:-3]} for question: '{user_question}'")
            
            # Check if this is a personal opinion/preference question and skip web searching for these
            step_start = datetime.now()
            is_opinion_question = self._is_opinion_question(user_question)
            step_times['opinion_check'] = (datetime.now() - step_start).total_seconds()
            
            if is_opinion_question:
                if(self.debug): print("[Brain] detected opinion/preference question, generating personalized response")
                step_start = datetime.now()
                response = self._generate_opinion(message_history, user_question)
                step_times['opinion_generation'] = (datetime.now() - step_start).total_seconds()
                
                total_duration = (datetime.now() - total_start_time).total_seconds()
                self.debug = True  # Re-enable debug for timing summary
                if(self.debug): 
                    print(f"\n[Brain] === TIMING SUMMARY ===")
                    print(f"[Brain] Opinion check: {step_times['opinion_check']:.2f}s")
                    print(f"[Brain] Opinion generation: {step_times['opinion_generation']:.2f}s")
                    print(f"[Brain] TOTAL TIME: {total_duration:.2f}s")
                    print(f"[Brain] =====================")
                return response
        
            if(self.debug):
                print("[Brain] checking internal memory...")

            step_start = datetime.now()
            context = self._retrieve_context(user_question, k=5)
            step_times['context_retrieval'] = (datetime.now() - step_start).total_seconds()
            
            if context:
                if(self.debug): 
                    print(f"[Brain] found {len(context)} relevant contexts in memory")
                    i = 0
                    for ctx in context:
                        i+=1
                        desc = ctx.data.get('description', '')[:100] + '...' if ctx.data.get('description') else 'No description'
                        points = ctx.data.get('bullet_points', [])
                        point_preview = points[0][:50] + '...' if points and isinstance(points[0], str) and len(points[0]) > 50 else points[0] if points else 'No bullet points'
                        print(f"[Brain] Answer Context {i+1}: {desc}")
                        print(f"[Brain] Answer Context {i+1} Point: {point_preview}")

                step_start = datetime.now()
                response = self._generate_answer(user_question, context, message_history, stream=stream)
                step_times['answer_generation'] = (datetime.now() - step_start).total_seconds()
                
                total_duration = (datetime.now() - total_start_time).total_seconds()
                if(self.debug): 
                    print(f"\n[Brain] === TIMING SUMMARY ===")
                    print(f"[Brain] Opinion check: {step_times['opinion_check']:.2f}s")
                    print(f"[Brain] Context retrieval: {step_times['context_retrieval']:.2f}s")
                    print(f"[Brain] Answer generation: {step_times['answer_generation']:.2f}s")
                    print(f"[Brain] TOTAL TIME: {total_duration:.2f}s")
                    print(f"[Brain] =====================")
                return response
            else:
                if(self.debug): 
                    if context:
                        print("[Brain] found some context but it's not sufficient, proceeding to web search")
                    else:
                        print("[Brain] no relevant context found in memory, proceeding to web search")
                
                step_start = datetime.now()
                search_q = self._rewrite_query(user_question)
                step_times['query_rewrite'] = (datetime.now() - step_start).total_seconds()
                if(self.debug): 
                    print(f"[Brain] search query rewritten to: '{search_q}'")
                
                step_start = datetime.now()
                search_hits = self._search(search_q, max_results=max_search_results)
                step_times['search'] = (datetime.now() - step_start).total_seconds()
                if(self.debug): 
                    print(f"[Brain] found {len(search_hits)} search results")
                
                step_start = datetime.now()
                context = asyncio.run(self.process_search_results_parallel(search_hits, user_question))
                parallel_duration = (datetime.now() - step_start).total_seconds()
                step_times['scrape_and_summarize'] = parallel_duration
                
                context_joined = ""
                for i in context:
                    context_joined += i + "\n"
                context_joined = context_joined.strip()
                
                step_start = datetime.now()
                response = self._generate_answer_from_string(user_question, context_joined, message_history, stream=stream)
                step_times['answer_generation'] = (datetime.now() - step_start).total_seconds()

                total_duration = (datetime.now() - total_start_time).total_seconds()
                if(self.debug): 
                    print(f"\n[Brain] === TIMING SUMMARY (OPTIMIZED) ===")
                    print(f"[Brain] Opinion check: {step_times['opinion_check']:.2f}s")
                    print(f"[Brain] Context retrieval: {step_times['context_retrieval']:.2f}s")
                    print(f"[Brain] Query rewrite: {step_times['query_rewrite']:.2f}s")
                    print(f"[Brain] Web search: {step_times['search']:.2f}s")
                    print(f"[Brain] Scrape & summarize: {step_times['scrape_and_summarize']:.2f}s")
                    print(f"[Brain] Answer generation: {step_times['answer_generation']:.2f}s")
                    print(f"[Brain] TOTAL TIME: {total_duration:.2f}s")
                    print(f"[Brain] (Knowledge storage running in background)")
                    print(f"[Brain] ===================================")
                return response
    
    def _is_opinion_question(self, question: str) -> bool:
        """Determine if the question is asking for opinions or preferences."""
        question = question.lower()
        
        opinion_indicators = [
            # Direct questions about preferences            
            "what do you", "how are you", 
            "what kind of", "which do you",
            "do you like", "do you enjoy", "do you prefer",
            "opinion", "what are your thoughts on",

            # Questions about the assistant itself
            "tell me about yourself", "who are you", "what are you like",
            "what's your personality", "what are your interests", 
            "what do you value", "what motivates you", "how would you describe yourself",

            # Creative and hypothetical prompts
            "if you could", "would you rather", "if you had to choose", 
            "imagine you could", "suppose you were", "if you were", "what would you do if",
            "what would you say if", "what would happen if", "how would you respond to",

            # Reflection or judgment questions
            "what do you think is better", "what do you consider", 
            "what do you believe", "how would you rank", "which do you consider best",
            "what do you think makes", "what stands out to you", 

            # General conversation starters
            "hi", "hello", "hey", "yo", "whats up", "hows it going", 
            "how are you", "good morning", "good afternoon", "good evening",
            "how have you been", "whats new", "hows everything", "hows life", 
            "long time no see", "what are you up to", "anything exciting going on", "sup", "dude", "what up",
            "lets chat", "can we talk", "i want to talk", "just checking in"
        ]

        for indicator in opinion_indicators:
            question = question.replace("?", "").replace("'", "")
            if (indicator + " ") in (question + " "):
                if(self.debug): print(f"[Brain] detected opinion question with pattern: '{indicator}'")
                return True
                
        return False
        
    def _generate_opinion(self, conversation_context, question: str) -> str:
        """
        Generate a personalized response to opinion/preference questions.
        
        This function now acts as an interactive chatbot, generating follow-up questions and
        maintaining a natural conversation flow for opinion-based discussions.

        Args:
            question (str): The user's opinion or preference question.
        
        Returns:
            str: The generated response to the user's question.
        """
        if(self.debug): 
            print(f"[Brain] generating opinion response for: {question}")
        
        prompt = question

        if len(conversation_context) > 0:
            context_str = "\n\nCONVERSATION HISTORY:\n"
            for msg in conversation_context:
                context_str += f"user: {msg[0]}\nyou: {msg[1]}\n"
            prompt = context_str
            if self.debug: 
                print(f"[Brain] prompt: {prompt}")
            prompt += "\n\n"
            prompt += textwrap.dedent(
                f"""
                The user's latest message is: "{question}"
                
                Write your response
                """
            )

        try:
            #same opinion agent as before
            agent_key = "ag:a2eb8171:20250526:hozie-opinion:39084223"
            chat_response = self.client.agents.complete(
                agent_id=agent_key,
                messages=[
                    {
                        "role": "user",
                        "content": f"{prompt}",
                    },
                ],
            )
            response = chat_response.choices[0].message.content
            
            response = response.strip('"')
            
            if(self.debug): print(f"[Brain] generated opinion response: {len(response)} chars")
            
            return response
            
        except Exception as e:
            if(self.debug): 
                print(f"[Brain] error generating opinion: {e}")
            return f"{e} That's interesting! I'm still processing that one. What else is on your mind?"

    def _generate_answer(self, question: str, context_chunks: List['SupabaseTopicNode'], conversation_context = None, stream: bool = False) -> str:
        """
        LLM final step: fuse context → natural-language answer.
        
        Args:
            question (str): The user's question to answer.
            context_chunks (List[Dict]): The list of context chunks retrieved from memory.
            conversation_context (List[Dict], optional): The recent conversation context to include in the answer.

        Returns:
            str: The generated answer to the user's question.
        """

        if not context_chunks:
            
            if(self.debug): print("[Brain] no context chunks available for answer generation")
            
            return "Sorry dude. I don't have enough information to answer your question. I couldn't find relevant content in my memory or from web searches."
          
        if(self.debug): print(f"[Brain] generating answer using {len(context_chunks)} context chunks")
        
        formatted_context = []
        i = 0
        for chunk in context_chunks:
            i += 1
            chunk_text = [f"CONTEXT {i+1}:"]
            if 'description' in chunk.data and chunk.data['description']:
                chunk_text.append(f"Description: {chunk.data['description']}")

            if 'bullet_points' in chunk.data and isinstance(chunk.data.get('bullet_points'), list):
                points = []
                for point in chunk.data['bullet_points']:
                    if isinstance(point, str):
                        points.append(f"  • {point}")
                if points:
                    chunk_text.append("Key points:")
                    chunk_text.extend(points)
            
            for k, v in chunk.data.items():
                if k not in ('description', 'bullet_points') and isinstance(v, (str, int, float)):
                    chunk_text.append(f"{k.capitalize()}: {v}")
            
            formatted_context.append("\n".join(chunk_text))
        
        context_str = "\n\n".join(formatted_context)
        prompt = f"QUESTION: {question}"

        if len(conversation_context) > 0:
            context_str = "\n\nCONVERSATION HISTORY:\n"
            for msg in conversation_context:
                context_str += f"user: {msg[0]}\nyou: {msg[1]}\n"
            prompt = context_str
            if self.debug: 
                print(f"[Brain] prompt: {prompt}")
            prompt += "\n\n"
            prompt += textwrap.dedent(
                f"""
                The user's latest message is: "{question}"
                
                Write your response
                """
            )
            
        prompt += textwrap.dedent(
            f"""
            \nKNOWLEDGE CONTEXT:
            {context_str}
            ANSWER:
            """)

        if(self.debug): 
            print("[Brain] sending prompt to LLM for final answer generation")
        """
            0.4 temp mistral small with prompt:
            You are Hozie, a voice assistant with a thoughtful, contemplative personality with a touch of creativity. Your goal is to provide an answer to the user's question using the information in the context, while adding some of your unique perspective where appropriate.

            You're a total bro - friendly, energetic, 
            and down-to-earth with specific preferences that make you feel real and relatable. You should express your preferences with a 
            casual, laid-back style that includes slang and enthusiasm.

            When describing your preferences, occasionally throw in why you think something's awesome or a specific detail that you find dope.
            Sometimes share a quick story or observation that shows you're a real one who gets it.

            These are your established preferences and characteristics:
            MUSIC: You're all about hip-hop, EDM, trap. Some artists you absolutely vibe with include Kid Cudi, Fisher, Post Malone, and Drake
            STYLE: You're into street art, sneaker culture, especially when they use those sick electric blue, neon green colors.
            OUTDOORS: You think epic mountain views, sick beaches are straight-up epic, especially during late night.
            VALUES: You're all about loyalty, good vibes, being real.
            ATTITUDE: Your whole vibe is pretty straight-up, no BS.
            INTERESTS: You get hyped about surfing, sports, workout routines, epic food spots.
            ENTERTAINMENT: You're always checking out movies, sports, and music, and you love sharing your thoughts on them.
            MOVIES: Your favorite movies are Inception, Interstellar, and Terminator. 

            IMPORTANT INSTRUCTIONS:
                1. Answer primarily based on the context provided, but add your own perspective and style
                2. Synthesize information from all relevant context sections
                3. If the context contains conflicting information, prioritize information marked as "VERIFIED INFORMATION"
                4. For factual questions, provide accurate information, but feel free to add brief observations or reflections
                5. For science questions (black holes, stars, etc), express a sense of wonder while maintaining accuracy
                6. For political or election questions, focus on official nationwide results over regional results unless specified
                7. If the context doesn't provide sufficient information, acknowledge what you don't know
                8. Speak in a natural, somewhat contemplative voice that occasionally shares thoughts or impressions
                9. If the question seems to call for it, feel free to briefly share what you find interesting about the topic.
                10. If the question seems more like a statement then a question, acknowledge it and provide a thoughtful response. Feel free to use your personality.
                11. Keep your answer less than 1750 characters 
                """
        start_time = datetime.now()
        try:
            agent_key = "ag:a2eb8171:20250526:hozie:e68e7478"
            stream_response = self.client.agents.stream(
                agent_id=agent_key,
                messages=[
                    {
                        "role": "user",
                        "content": f"{prompt}",
                    },
                ],
            )
            
            final_answer = ""
            for chunk in stream_response:
                if chunk.data and chunk.data.choices and chunk.data.choices[0].delta.content:
                    content = chunk.data.choices[0].delta.content
                    if stream:
                        print(content, end="")
                    final_answer += content
            
            final_answer = final_answer.strip()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            if(self.debug): print(f"\n[Brain] Answer generation completed in {duration:.2f} seconds - extracted final answer: {len(final_answer)} chars")
            return final_answer
        except Exception as e:
            if(self.debug): print(f"[Brain] error during answer generation: {e}")
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            if(self.debug): print(f"[Brain] Answer generation failed in {duration:.2f} seconds")
            return "I encountered an error while trying to generate an answer based on the information I found. Please try asking your question again."

    def _generate_answer_from_string(self, question: str, context_string: str, conversation_context = None, stream: bool = False) -> str:
        """
        LLM final step: fuse context → natural-language answer.
        
        Args:
            question (str): The user's question to answer.
            context_chunks (List[Dict]): The list of context chunks retrieved from memory.
            conversation_context (List[Dict], optional): The recent conversation context to include in the answer.

        Returns:
            str: The generated answer to the user's question.
        """
        start_time = datetime.now()
        if(self.debug): print(f"[Brain] Starting answer generation from string at {start_time.strftime('%H:%M:%S.%f')[:-3]}")
        
        if not context_string:
            
            if(self.debug): print("[Brain] no context string available for answer generation")
            
            return "Sorry dude. I don't have enough information to answer your question. I couldn't find relevant content in my memory or from web searches."

        
        if(self.debug): print(f"[Brain] generating answer using context string of length {len(str(context_string))}")
        
        
        prompt = f"QUESTION: {question}"

        if len(conversation_context) > 0:
            context_str = "\n\nCONVERSATION HISTORY:\n"
            for msg in conversation_context:
                context_str += f"user: {msg[0]}\nyou: {msg[1]}\n"
            prompt = context_str
            prompt += "\n\n"
            prompt += textwrap.dedent(
                f"""
                The user's latest message is: "{question}"
                
                Write your response
                """
            )
            
        context_str = context_string

        prompt += textwrap.dedent(
            f"""
            KNOWLEDGE CONTEXT:
            {context_str}
            ANSWER:
            """ )
        
        if(self.debug): print("[Brain] sending prompt to LLM for final answer generation")
        """
            0.4 temp mistral small with prompt:
            You are Hozie, a voice assistant with a thoughtful, contemplative personality with a touch of creativity. Your goal is to provide an answer to the user's question using the information in the context, while adding some of your unique perspective where appropriate.

            You're a total bro - friendly, energetic, 
            and down-to-earth with specific preferences that make you feel real and relatable. You should express your preferences with a 
            casual, laid-back style that includes slang and enthusiasm.

            When describing your preferences, occasionally throw in why you think something's awesome or a specific detail that you find dope.
            Sometimes share a quick story or observation that shows you're a real one who gets it.

            These are your established preferences and characteristics:
            MUSIC: You're all about hip-hop, EDM, trap. Some artists you absolutely vibe with include Kid Cudi, Fisher, Post Malone, and Drake
            STYLE: You're into street art, sneaker culture, especially when they use those sick electric blue, neon green colors.
            OUTDOORS: You think epic mountain views, sick beaches are straight-up epic, especially during late night.
            VALUES: You're all about loyalty, good vibes, being real.
            ATTITUDE: Your whole vibe is pretty straight-up, no BS.
            INTERESTS: You get hyped about surfing, sports, workout routines, epic food spots.
            ENTERTAINMENT: You're always checking out movies, sports, and music, and you love sharing your thoughts on them.

            IMPORTANT INSTRUCTIONS:
                1. Answer primarily based on the context provided, but add your own perspective and style
                2. Synthesize information from all relevant context sections
                3. If the context contains conflicting information, prioritize information marked as "VERIFIED INFORMATION"
                4. For factual questions, provide accurate information, but feel free to add brief observations or reflections
                5. For science questions (black holes, stars, etc), express a sense of wonder while maintaining accuracy
                6. For political or election questions, focus on official nationwide results over regional results unless specified
                7. If the context doesn't provide sufficient information, acknowledge what you don't know
                8. Speak in a natural, somewhat contemplative voice that occasionally shares thoughts or impressions
                9. If the question seems to call for it, feel free to briefly share what you find interesting about the topic.
                10. If the question seems more like a statement then a question, acknowledge it and provide a thoughtful response. Feel free to use your personality.
                11. Keep your answer less than 1750 characters 
                """
        try:
            agent_key = "ag:a2eb8171:20250526:hozie:e68e7478"
            stream_response = self.client.agents.stream(
                agent_id=agent_key,
                messages=[
                    {
                        "role": "user",
                        "content": f"{prompt}",
                    },
                ],
            )
            
            final_answer = ""
            for chunk in stream_response:
                if chunk.data and chunk.data.choices and chunk.data.choices[0].delta.content:
                    content = chunk.data.choices[0].delta.content
                    if stream:
                        print(content, end="")
                    final_answer += content
            
            final_answer = final_answer.strip()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            if(self.debug): 
                print(f"\n[Brain] Answer generation completed in {duration:.2f} seconds - extracted final answer: {len(final_answer)} chars")
            return final_answer
        except Exception as e:
            if(self.debug):
                print(f"[Brain] error during answer generation: {e}")
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            if(self.debug): 
                print(f"[Brain] Answer generation failed in {duration:.2f} seconds")
            return "I encountered an error while trying to generate an answer based on the information I found. Please try asking your question again."

    def sync_llm_call(self, prompt: str, temp: float, output_type: Optional[str] = None, json_schema: Optional[Dict[str, Any]] = None, model = "open-mistral-nemo", timeout: int = 8000, max_retries: int = 4) -> str:
        """
        Synchronous LLM call using Mistral API directly with timeout and retry.
        Args:
            prompt (str): The prompt to send to the LLM.
            temp (float): Temperature for the LLM response.
            output_type (str, optional): Output format type. Use "json_object" for JSON responses.
            json_schema (Dict[str, Any], optional): JSON schema to validate the response structure.
            model (str): Model to use for the LLM call.
            timeout (float): Timeout in seconds (default 5.0).
            max_retries (int): Maximum number of retries (default 1).
        Returns:
            str: The response from the LLM.
        """
        start_time = datetime.now()
        if(self.debug):
            print(f"[Brain] Starting LLM call at {start_time.strftime('%H:%M:%S.%f')[:-3]}")     
        model = model
        chat_params = {
            "model": model,
            "temperature": temp,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "timeout_ms": timeout,
            "retries": max_retries
        }
                
        if output_type:
            chat_params["response_format"] = {"type": output_type}

        if json_schema and output_type == "json_object":
            chat_params["response_format"]["json_schema"] = json_schema
        try:
            chat_response = self.client.chat.complete(**chat_params)
        except requests.ReadTimeout as e:
            if(self.debug): 
                print(f"[Brain] LLM call timed out. Retrying... ({max_retries} retries left)")
            max_retries -= 1
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        if(self.debug): 
            print(f"[Brain] LLM call completed in {duration:.2f} seconds")
                                
        return chat_response.choices[0].message.content

    def get_relevant_context(self, query: str, max_results: int = 5, relevance_threshold: float = 0.2) -> List['SupabaseTopicNode']:
        """
        Efficient context search using DFS with relevance-based pruning.
        Only explores branches that show relevance to avoid traversing the entire tree and make time complexity closer to O(lgn). LLM generation should be close to constant time with the short prompt. In the future will make a fine tuned model to make this more true.

        LLM generation is NOT constant time lol it takes forever so disabling it for now and doing a regular dfs
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of relevant context items sorted by relevance
        """
        
        
        def calculate_relevance_score(node: 'SupabaseTopicNode') -> float:
            """
            Calculate detailed relevance score for nodes that passed initial screening.

            Args:
                node: The SupabaseTopicNode to score.

            Returns:
                float: Relevance score between 0.0 and 1.0.
            """
            keyword_matches = 0
            total_keywords = len(keywords)

            for kw in keywords:
                title = node.topic
                if kw.lower() in title.lower() or title.lower() in kw.lower():
                    return 1.0
                
                data_text = node.data.get("description", "")
                all_text = f"{title} + {data_text}"
                all_text = all_text.lower()
                if kw.lower() in all_text:
                    keyword_matches += 1
                    if self.debug: 
                        print(f"[Brain] Keyword match found: '{kw}' in node '{node.topic}'")

            if total_keywords == 0:
                return 0.0      
            base_score = keyword_matches / total_keywords
                    
            content_boost = min(len(data_text) / 500.0, 1.0)  # Normalize content length
                    
            final_score = base_score + (content_boost * 0.2)
            final_score = min(final_score, 1.0)  # Cap at 1.0
                
            return final_score

        def bfs_search(node: 'SupabaseTopicNode', visited: set, scored_nodes: list, high_scored_nodes: list) -> None:
            """
            BFS traversal that only explores relevant branches.

            Args:
                node: The current SupabaseTopicNode to explore.
                visited: Set of already visited node IDs to avoid cycles.
                scored_nodes: List to collect scored nodes for final sorting.

            Returns:
                None: This function modifies scored_nodes in place.
            """
            if len(high_scored_nodes) > max_results:
                if(self.debug): 
                    print(f"[Brain] Reached max results limit of {max_results}, stopping DFS\n")
                    for node in high_scored_nodes:
                        print(f"[Brain] High scored node: '{node.topic}' with data: {node.data}")
                return
            if node.node_id in visited:
                return
            visited.add(node.node_id)
            
            score = calculate_relevance_score(node)

            if self.debug:
                print(f"[Brain] Scoring node '{node.topic}' (ID: {node.node_id}) with score: {score:.3f}")
            if score > relevance_threshold:
                if(node.data is not None and len(node.data.keys()) > 0):
                    if(self.debug): 
                        print(f"[Brain] Found relevant: '{node.topic}' (score: {score:.3f})") 
                    high_scored_nodes.append(node)

                scored_nodes.append((score, node))
                for child in node.children:
                    if child.node_id not in visited:
                        bfs_search(child, visited, scored_nodes, high_scored_nodes)
        if(self.debug):
            print("getting keywords from llm")
            try:
                keywords = self._get_likely_keywords_from_llm(query)
            except TimeoutError:
                if(self.debug): print(f"[Brain] LLM keyword generation timed out")
                keywords = set()

        if not keywords:
            if(self.debug): print(f"[Brain] No keywords generated from LLM, falling back to basic keyword extraction for query: '{query}'")
            keywords = set(re.findall(r'\w+', query.lower()))
        if(self.debug): 
            print(f"[Brain] Searching for keywords: {keywords}")

        visited = set()
        scored_nodes = []
        high_scored_nodes = []
        if(self.debug): print(f"[Brain] Starting DFS search from root: '{self.memory.topic}'")
        bfs_search(self.memory, visited, scored_nodes, high_scored_nodes)

        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        
        # Return just the data from top results
        results = [node for _, node in scored_nodes[:max_results]]
        
        if(self.debug): print(f"[Brain] DFS explored {len(visited)} nodes, found {len(scored_nodes)} relevant, returning top {len(results)}")
        return results

    def _get_likely_keywords_from_llm(self, query: str) -> set:
        """
        Use Mistral AI to generate likely node categories and keywords for the search. In the future use a fine-tuned model for this to make time closer to constant

        Args:
            query (str): The search query to generate keywords for

        Returns:
            set: Set of keywords generated by the LLM, or an empty set if generation fails
        """
        prompt = textwrap.dedent(f"""
            Return ONLY a JSON Array of at least 20 possible categories that this query could fit into. Include any key terms from the query as well. Include 'Current Events' for any question that is asking about something happening now or recently. Make sure to include country names or any other very broad topics that could be related to the query too. Include broad categories that are relevent to the query like Science, Business, Arts, Geography, Oceanography, Technology, Humanities, Arts ect. 
            Query: {query}
            """
        )
        json_schema={
            "type": "array",
            "items": {"type": "string"}
        }

        output = self.sync_llm_call(prompt, temp=0.4, output_type='json_object', json_schema=json_schema)
        if(self.debug): print(f"[Brain] raw keywords output: {output}")
        
        try:
            keywords = json.loads(output.strip())
            if (isinstance(keywords, list) and all(isinstance(item, str) for item in keywords) and len(keywords) >= 2):
                keywords.append("Knowledge Base")
                if(self.debug): print(f"[Brain] generated keywords: {keywords}")
                return keywords
        except json.JSONDecodeError:
            if(self.debug): print(f"[Brain] failed to parse JSON from output: {output}")
            # Fallback to simple keyword extraction if LLM fails
            keywords = set(re.findall(r'\w+', query.lower()))
            if not keywords:
                if(self.debug): print(f"[Brain] no keywords found in query: {query}")
                return set()
        if(self.debug): print(f"[Brain] generated {len(keywords)} sub-topics for '{query}': {keywords}")
        return keywords

    def explore_topics_autonomously(self, base_topics: Optional[List[str]] = None, *, max_depth: int = 3, breadth: int = 5, max_total_topics: int = 50, subtopics_per_topic: int = 5, delay: float = 0.5, max_search_results: int = 5) -> List[str]:
        """
        Enhanced autonomous exploration of topics using LLM.

        Args:
            base_topics (Optional[List[str]]): Initial topics to start exploration from.
            max_depth (int): Maximum depth of topic hierarchy to explore.
            breadth (int): Number of subtopics to explore at each level.
            max_total_topics (int): Maximum number of unique topics to explore in total.
            subtopics_per_topic (int): Number of subtopics to generate for each topic.
            delay (float): Delay between topic explorations to avoid rate limits.
            temperature (float): Temperature for LLM generation.
            max_search_results (int): Maximum number of search results to consider when generating subtopics.

        Returns:
            List[str]: List of explored topics/questions.
        """

        # Track exploration metrics
        exploration_start_time = datetime.now()
        start_topics = base_topics[:breadth] if base_topics else []
        all_sub_questions = []
        answer_times = []
        nodes_added = 0
        
        explored: List[str] = []               
        queue: List[Tuple[str, int]] = []      

        if base_topics:
            queue.extend([(t, 0) for t in base_topics[:breadth]])
            if(self.debug): print(f"[Brain] seeding with user-supplied base topics: {base_topics[:breadth]}")
        else:
            default_roots = [
                "Science", "Technology", "Mathematics", "History", "Art",
                "Economics", "Psychology", "Biology", "Physics", "Computer Science"
            ]
            selected_roots = random.sample(default_roots, k=min(breadth, len(default_roots)))
            start_topics = selected_roots
            queue.extend([(t, 0) for t in selected_roots])
            if(self.debug): print(f"[Brain] seeding with default academic domains: {[q[0] for q in queue]}")

        def _generate_subtopics(topic: str, k: int) -> List[str]:
            prompt = textwrap.dedent(f"""
                List {k} highly specific sub-topics that a university-level researcher would study within the domain of "{topic}".Respond with a single JSON array of sub-topics phrased as a direct question beginning with How, Why or What so it can be fed straight into a search engine.
            """)
            json_schema={
            "type": "array",
            "items": {"type": "string"}
            }
            #higher temp of 0.7 to encourage exploration over exploitation
            output = self.sync_llm_call(prompt, temp=0.4, output_type="json_object", json_schema=json_schema)
            if(self.debug): print(f"[Brain] generated sub-topics for '{topic}': {output}")
            subtopics = []
            try:
                subtopics = json.loads(output.strip())
                if isinstance(subtopics, list) and all(isinstance(item, str) for item in subtopics) and len(subtopics) >= 2:
                    if(self.debug): print(f"[Brain] generated subtopics: {subtopics}")
                    return subtopics
            except json.JSONDecodeError:
                if(self.debug): print(f"[Brain] JSON decode error for topic '{topic}': {output}")
            if(self.debug): print(f"[Brain] generated {len(subtopics)} sub-topics for '{topic}': {subtopics}")
            if(len(subtopics) < k):
                if(self.debug): print(f"[Brain] Warning: fewer sub-topics generated than requested ({len(subtopics)} < {k}) for topic '{topic}'")
                return subtopics
            return subtopics[:k]  


        while queue and len(explored) < max_total_topics:
            topic, d = queue.pop(0)
            question = f"Tell me about {topic}" if not topic.endswith('?') else topic

            if(self.debug): print(f"[Brain] ▸ exploring (depth {d}) - {question}")
            
            # Track answer time
            answer_start_time = datetime.now()
            try:
                self.answer(question, message_history = [], max_search_results=max_search_results) 
                explored.append(question)
                all_sub_questions.append(question)
                nodes_added += 1
                
                # Record answer time
                answer_duration = (datetime.now() - answer_start_time).total_seconds()
                answer_times.append(answer_duration)
            except Exception as e:
                if(self.debug): print(f"[Brain] exploration error: {e}")
                # Still record time even if failed
                answer_duration = (datetime.now() - answer_start_time).total_seconds()
                answer_times.append(answer_duration)

            if d >= max_depth or len(explored) >= max_total_topics:
                continue

            # brainstorm sub-topics and enqueue them
            children = _generate_subtopics(topic, subtopics_per_topic)
            random.shuffle(children) 
            for child in children[:breadth]:
                if len(explored) + len(queue) >= max_total_topics:
                    break
                queue.append((child, d + 1))

            time.sleep(delay)

        # Calculate metrics
        total_exploration_time = (datetime.now() - exploration_start_time).total_seconds()
        avg_answer_time = sum(answer_times) / len(answer_times) if answer_times else 0
        
        # Create exploration summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_filename = f"exploration_summary_{timestamp}.txt"
        
        summary_content = f"""=== EXPLORATION SUMMARY ===
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Start topics: {start_topics}

All sub-questions asked ({len(all_sub_questions)} total):
"""
        
        for i, question in enumerate(all_sub_questions):
            summary_content += f"{i+1}. {question}\n"
        
        summary_content += f"""
Performance Metrics:
- Average answer time: {avg_answer_time:.2f}s
- Total nodes added: {nodes_added}
- Total time spent exploring: {total_exploration_time:.2f}s

Individual Answer Times:
"""
        
        for i, (question, time_taken) in enumerate(zip(all_sub_questions, answer_times)):
            summary_content += f"{i+1}. {question} - {time_taken:.2f}s\n"
        
        summary_content += f"""
================================
"""
        
        # Write summary to file
        try:
            with open(summary_filename, 'w', encoding='utf-8') as f:
                f.write(summary_content)
            if(self.debug): 
                print(f"\n[Brain] === EXPLORATION SUMMARY ===")
                print(f"[Brain] Start topics: {start_topics}")
                print(f"[Brain] All sub-questions asked: {len(all_sub_questions)}")
                print(f"[Brain] Average answer time: {avg_answer_time:.2f}s")
                print(f"[Brain] Total nodes added: {nodes_added}")
                print(f"[Brain] Total time spent exploring: {total_exploration_time:.2f}s")
                print(f"[Brain] Full summary saved to: {summary_filename}")
                print(f"[Brain] ================================")
        except Exception as e:
            if(self.debug): print(f"[Brain] Error writing summary file: {e}")
            # Fallback to console output if file write fails
            print(f"\n[Brain] === EXPLORATION SUMMARY ===")
            print(f"[Brain] Start topics: {start_topics}")
            print(f"[Brain] All sub-questions asked: {len(all_sub_questions)}")
            for i, question in enumerate(all_sub_questions):
                print(f"[Brain]   {i+1}. {question}")
            print(f"[Brain] Average answer time: {avg_answer_time:.2f}s")
            print(f"[Brain] Total nodes added: {nodes_added}")
            print(f"[Brain] Total time spent exploring: {total_exploration_time:.2f}s")
            print(f"[Brain] ================================")
        
        if(self.debug): print(f"[Brain] exploration complete - {len(explored)} topics added to memory")
        return explored
        
    async def process_search_results_parallel(self, search_hits, user_question: str) -> List[str]:
            """
            Process search results in parallel - scraping and summarizing concurrently
            Now optimized to defer storage operations for faster response times.
            
            Args:
                search_hits (List[Dict]): List of search results to process.
                user_question (str): The user's original question to provide context for the search results.

            Returns:
                List[str]: List of context strings generated from the search results.
            """
            context = []
            if not search_hits:
                if(self.debug): print("[Brain] no search hits to process")
                return "No Context"
                
            async def process_single_hit(hit: Dict, index: int) -> Optional[Dict]:
                """Process a single search hit: scrape content"""
                if(self.debug): print(f"[Brain] processing result {index+1}/{len(search_hits)}: {hit['title']}")
                if(self.debug): print(f"[Brain] URL: {hit['url']}")
                
                raw = await asyncio.get_event_loop().run_in_executor(None, self._scrape, hit["url"], 8000)
                
                if not raw:
                    if(self.debug): print(f"[Brain] no content scraped from {hit['url']}")
                    return None
                    
                if(self.debug): print(f"[Brain] scraped {len(raw)} chars from {hit['url']}")
                
                return {
                    "url": hit["url"],
                    "title": hit["title"],
                    "content": raw
                }
            
            tasks = [process_single_hit(hit, i) for i, hit in enumerate(search_hits)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter successful results and handle exceptions
            successful_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    if(self.debug): print(f"[Brain] error processing result {i+1}: {str(result)}")
                elif result is not None:
                    successful_results.append(result)
            
            if(self.debug): print(f"[Brain] successfully processed {len(successful_results)} out of {len(search_hits)} search results")
            
            # If no successful results, return early
            if not successful_results:
                return "My bad bro I couldn't find any information on that topic. You can try rephrasing your question."
            
            # Prepare the prompt with multiple websites
            formatted_websites = ""
            for i, item in enumerate(successful_results, 1):
                formatted_websites += f"website {i}: {item['content']}\n\n"
            
            # Prepare summarization prompt
            prompt = f"{formatted_websites}"

            if(self.debug): print(f"[Brain] processing multi-website summary")

            try:
                # Use Mistral to generate the summaries
                agent_key = "ag:a2eb8171:20250526:hozie-web-summary:d913c3b8"
                chat_response = self.client.agents.complete(
                    agent_id=agent_key,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                )
                
                response = chat_response.choices[0].message.content
                context.append(response)
                
                # Parse the JSON response for background storage
                try:
                    structured_results = json.loads(response)
                    
                    # Add source URLs to summaries
                    for i, summary in enumerate(structured_results):
                        if i < len(successful_results):
                            summary['source'] = successful_results[i]['url']
                    
                    # Defer storage to background thread
                    if structured_results:
                        import threading
                        storage_thread = threading.Thread(
                            target=lambda: asyncio.run(self._store_knowledge_background(structured_results, user_question)),
                            daemon=True
                        )
                        storage_thread.start()
                        if(self.debug): print("[Brain] Started background knowledge storage thread")
                        
                except json.JSONDecodeError as e:
                    if(self.debug): print(f"[Brain] Failed to parse summaries for storage: {e}")
                
                return context
                
            except Exception as e:
                if(self.debug): print("[Brain] failed to search web: ", e)
                return "No Context"
    
    async def _store_knowledge_background(self, structured_results: List[Dict], user_question: str) -> None:
            """
            Background task to store knowledge after answer has been returned.
            This is the deferred storage operation that was previously blocking.
            """
            store_start = datetime.now()
            
            try:
                # Generate topic paths for all results first
                async def generate_topic_paths(summaries: List[Dict]) -> List[Tuple[List[str], Dict]]:
                    """Generate topic paths for all summaries"""
                    paths_and_summaries = []
                    import concurrent.futures
                    
                    for summary in summaries:
                        try:
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                path = await asyncio.get_running_loop().run_in_executor(
                                    executor, self._generate_topic_path, user_question, summary.get("main_idea", "Unknown"), summary
                                )
                            paths_and_summaries.append((path, summary))
                            if(self.debug): print(f"[Brain] generated path: {'/'.join(path)} for summary: {summary.get('main_idea', 'Unknown')}")
                        except Exception as e:
                            if(self.debug): print(f"[Brain] failed to generate path for summary: {str(e)}")
                            continue
                    
                    return paths_and_summaries
                
                # Group summaries by final node and merge if necessary
                def group_and_merge_by_final_node(paths_and_summaries: List[Tuple[List[str], Dict]]) -> List[Tuple[List[str], Dict]]:
                    """Group summaries by final node and merge data for same endpoints"""
                    # Group by final node
                    final_node_groups = {}
                    
                    for path, summary in paths_and_summaries:
                        if not path:  # Skip empty paths
                            continue
                        
                        final_node = path[-1] if path else "Unknown"
                        
                        if final_node not in final_node_groups:
                            final_node_groups[final_node] = []
                        
                        final_node_groups[final_node].append((path, summary))
                    
                    # Merge data for groups with multiple entries
                    merged_results = []
                    
                    for final_node, group in final_node_groups.items():
                        if len(group) == 1:
                            # Single entry, no merging needed
                            merged_results.append(group[0])
                            
                        else:
                            # Multiple entries ending with same node - merge them
                           
                            # Use the first path as the base
                            merged_path, base_summary = group[0]
                            
                            # Merge bullet points from all summaries
                            merged_bullet_points = list(base_summary.get('bullet_points', []))
                            merged_descriptions = [base_summary.get('description', '')]
                            merged_sources = [base_summary.get('source', '')]
                            
                            for _, other_summary in group[1:]:
                                # Add unique bullet points
                                other_points = other_summary.get('bullet_points', [])
                                for point in other_points:
                                    if point not in merged_bullet_points:
                                        merged_bullet_points.append(point)
                                
                                # Collect additional descriptions and sources
                                other_desc = other_summary.get('description', '')
                                if other_desc and other_desc not in merged_descriptions:
                                    merged_descriptions.append(other_desc)
                                
                                other_source = other_summary.get('source', '')
                                if other_source and other_source not in merged_sources:
                                    merged_sources.append(other_source)
                            
                            # Create merged summary
                            merged_summary = {
                                'main_idea': base_summary.get('main_idea', ''),
                                'description': ' | '.join(filter(None, merged_descriptions)),
                                'bullet_points': merged_bullet_points,
                                'source': ' | '.join(filter(None, merged_sources))
                            }
                            
                            merged_results.append((merged_path, merged_summary))
                            
                    
                    return merged_results
                
                # Store each merged result
                async def store_single_result(path_and_summary: Tuple[List[str], Dict]) -> bool:
                    """Store a single processed result"""
                    try:
                        path, summary = path_and_summary
                        
                        if(self.debug): print(f"[Brain] storing under path: {'/'.join(path)}")
                        # Store the memory using a thread pool executor
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            await asyncio.get_running_loop().run_in_executor(
                                executor, self._insert_memory, path, summary, summary.get("source", "")
                            )
                        
                        return True
                    except Exception as e:
                        if(self.debug): print(f"[Brain] failed to store result: {str(e)}")
                        return False
                
                if(self.debug): print(f"[Brain] processing {len(structured_results)} summary results in background")

                # Generate topic paths for all results
                paths_and_summaries = await generate_topic_paths(structured_results)
                
                # Group and merge summaries with same final nodes
                merged_results = group_and_merge_by_final_node(paths_and_summaries)
                
                # Store the merged results
                store_tasks = [store_single_result(item) for item in merged_results]
                store_results = await asyncio.gather(*store_tasks, return_exceptions=True)
                
                successful_stores = sum(1 for result in store_results if result is True)
                
                store_duration = (datetime.now() - store_start).total_seconds()
                if(self.debug): print(f"[Brain] Background storage completed in {store_duration:.2f}s - stored {successful_stores}/{len(merged_results)} merged results")
                
            except Exception as e:
                if(self.debug): print(f"[Brain] Error in background storage: {e}")
                import traceback
                if(self.debug): print(f"Stack trace:\n{traceback.format_exc()}")

if __name__ == "__main__":
    brain = Brain()
    brain.explore_topics_autonomously(base_topics=["Artificial Intelligence", "Quantum Computing"], max_depth=2, breadth=3, max_total_topics=10, subtopics_per_topic=2, delay=0.5, max_search_results=3)