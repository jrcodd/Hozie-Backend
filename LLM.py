#Using ministral causes a lot of errors when trying to get a specific json output even if it is a little faster. Useing Nemo because it can do it better.
import json
import os
import re
import textwrap
import time
import random
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence, Tuple, Optional

import requests
from bs4 import BeautifulSoup
from mistralai import Mistral, UserMessage

from supabase_topic_node import SupabaseTopicNode 
from supabase_chat_history import SupabaseChatHistory 

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

    def _insert_memory(self, topic_path: Sequence[str], data: Dict, source_url: str, confidence: float = 0.80) -> None:
        """
        Create/merge SupabaseTopicNodes along topic_path and attach data.
        
        Args:
            topic_path: List of topics representing the path in the memory tree.
            data: Dictionary containing the data to store at the leaf node.
            source_url: URL of the source where the data was retrieved from.
            confidence: Confidence level of the data (default is 0.80).            
        """
        
        if not topic_path:
            if(self.debug): print("[Brain] ERROR: Empty topic path provided, using fallback path")
            topic_path = ["Uncategorized", "WebContent"]
        
        # Ensure topic_path is a list of strings
        if not isinstance(topic_path, (list, tuple)):
            if(self.debug): print(f"[Brain] ERROR: topic_path is not iterable: {type(topic_path)}, converting to list")
            topic_path = [str(topic_path)]
        elif not all(isinstance(item, str) for item in topic_path):
            if(self.debug): print("[Brain] ERROR: topic_path contains non-string items, converting")
            topic_path = [str(item) for item in topic_path]
            
        if not data:
            if(self.debug): print("[Brain] ERROR: Empty data provided, skipping memory insertion")
            return
            
        if(self.debug): print(f"[Brain] inserting data into memory at path: {'/'.join(topic_path)}")
        
        try:
            if topic_path[0] == self.memory.topic:
                found_similar, existing_node, existing_path = self._check_similar_path(self.memory, topic_path[1:])
            else:
                found_similar, existing_node, existing_path = self._check_similar_path(self.memory, topic_path)
            
            if found_similar:
                if(self.debug): print(f"[Brain] found similar existing path: {'/'.join(existing_path)}")
                current_node = existing_node
            else:
                if(self.debug): print(f"[Brain] creating new path: {'/'.join(topic_path)}")
                
                current_node = self.memory
                for i, topic in enumerate(topic_path):
                    child = current_node.find_child(topic)
                    if not child:
                        if i == len(topic_path) - 1:
                            child = current_node.add_child(topic, data)
                            if(self.debug): print(f"[Brain] created leaf node: {child.topic}")
                        else:
                            child = current_node.add_child(topic)
                            if(self.debug): print(f"[Brain] created intermediate node: {child.topic}")
                    current_node = child
            
            node = current_node
            if(self.debug): print(f"[Brain] node created/found: {node.topic}")
                        
            # Merge dictionaries newer values overwrite older ones
            if isinstance(node.data, dict) and node.data:
                if(self.debug): print(f"[Brain] merging with existing data ({len(node.data)} keys)")
                node.data.update(data)
            else:
                if(self.debug): print("[Brain] setting new data")
                node.data = data

            meta_entry = {
                "url": source_url,
                "timestamp":  _get_time(),
                "confidence": confidence,
            }
            node.metadata.setdefault("sources", []).append(meta_entry)
            node.metadata["last_updated"] =  _get_time()
            if(self.debug): print(f"[Brain] successfully stored data under: {'/'.join(topic_path)}")
                        
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
            1. Starts with one of these broad domains only: Science, Humanities, Technology, Arts, Social Sciences
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
        output = self.sync_llm_call(prompt, temp=0.4, output_type="json_object", json_schema=json_schema)
        if(self.debug): print(f"[Brain] topic path generation output: {output}")
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
            node_count = len([self.memory] + self.memory.get_all_descendants())
            
            if node_count < k:
                if(self.debug): print("[Brain] memory tree is empty or nearly empty using web search")
                return []
            
            contexts = self.get_relevant_context(query, max_results=k*3)

            if(self.debug): print(f"[Brain] found {len(contexts)} context chunks")

            if contexts:
                unique_contexts = []
                description_set = set()   

                for ctx in contexts:
                    if ctx.get('description') and ctx.get('bullet_points'):
                        desc = ctx.get('description', '')
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

    def _retrieve_recent_context(self, user: str, k: int = 5) -> List[Dict]:
        """
        Retrieve recent context from chat history.
        This method retrieves the last k messages from the chat history to provide context for follow-up questions.

        Args:
            user_question: The user's question to analyze.
            k: The number of recent messages to retrieve (default is 5).
        
        Returns:
            A list of dictionaries containing recent context chunks from the chat history.
        """
        if(self.debug): print(f"[Brain] retrieving recent context for user: '{user}'")
        
        if not self.chat_history:
            self.chat_history = SupabaseChatHistory()
        
        recent_context = self.chat_history.get_recent_messages(k)
        
        if(self.debug): print(f"[Brain] found {len(recent_context)} recent messages")
        
        return recent_context
    def answer(self, user_question: str, max_search_results: int = 5) -> str:
        """
        Main entry-point: answer a user question using RAG.
        Args:
            user_question (str): The user's question to answer
            max_search_results (int, optional): Maximum number of websites to search when looking for information. Defaults to 5.
        Returns:
            str: The generated answer to the user's question.
        """
        if(self.debug): print(f"[Brain] processing question: '{user_question}'")
        
        # Check if this is a personal opinion/preference question and skip web searching for these
        is_opinion_question = self._is_opinion_question(user_question)
        if is_opinion_question:
            if(self.debug): print("[Brain] detected opinion/preference question, generating personalized response")
            response = self._generate_opinion(user_question)
            return response
        
        #TODO: add previous conversation to the context if this is a follow-up question
        is_followup = self._is_followup_question(user_question)
        recent_context = []

        if is_followup:
            # access supabase_chat_history to get the recent context
            if(self.debug): print("[Brain] detected follow up question, retrieving recent context")
            recent_context = self._retrieve_recent_context(user_question)
        
        if(self.debug): print("[Brain] checking internal memory...")
        context = self._retrieve_context(user_question, k=5)
        if context:
            if(self.debug): print(f"[Brain] found {len(context)} relevant contexts in memory")
            if(self.debug): print("[Brain] internal context sufficient.")
            
            #debug print context
            for i, ctx in enumerate(context):
                desc = ctx.get('description', '')[:100] + '...' if ctx.get('description') else 'No description'
                points = ctx.get('bullet_points', [])
                point_preview = points[0][:50] + '...' if points and isinstance(points[0], str) and len(points[0]) > 50 else points[0] if points else 'No bullet points'
                if(self.debug): print(f"[Brain] Answer Context {i+1}: {desc}")
                if(self.debug): print(f"[Brain] Answer Context {i+1} Point: {point_preview}")

            response = self._generate_answer(user_question, context, recent_context)
            return response
        else:
            if context:
                if(self.debug): print("[Brain] found some context but it's not sufficient, proceeding to web search")
            else:
                if(self.debug): print("[Brain] no relevant context found in memory, proceeding to web search")
            
            search_q = self._rewrite_query(user_question)
            if(self.debug): print(f"[Brain] search query rewritten to: '{search_q}'")
            
            search_hits = self._search(search_q, max_results=max_search_results)
            if(self.debug): print(f"[Brain] found {len(search_hits)} search results")
            
            # Capture the return value from the async function

            context = asyncio.run(self.process_search_results_parallel
            (search_hits, user_question))
            context_joined = ""
            for i in context:
                context_joined += i + "\n"
            context_joined = context_joined.strip()
            response = self._generate_answer(user_question, context_joined, recent_context)

            return response

    def _is_followup_question(self, question: str) -> bool:
        """
        Determine if the question is a follow-up to a previous conversation.
        
        Args:
            question (str): The user's question to analyze.
        
        Returns:
            bool: True if the question is likely a follow-up, False otherwise.
        """
            
        # Look for indicators of follow-up questions
        followup_indicators = [
            # Pronouns that refer to something previously mentioned
            "they", "them", "those", "these", "this",
            
            # Direct references to previous information
            "as mentioned", "as you said", "as we discussed", "earlier", "previous",
            "you just said", "you told me", "you mentioned",
            
            # Questions that build on previous context
            "and what about", "what else", "tell me more", "can you elaborate",
            "why is that", "how does that work", "could you explain", 
            "what if", "is there more", "go on", "continue", "furthermore",
            
            # Short standalone questions that rely on context
            "what if", "which of those", 
            
            # Direct continuations
            "also", "additionally", "moreover"
        ]
        
        # Very short questions are often follow-ups
        if len(question.split()) <= 3 and not question.startswith("what is") and not question.startswith("how to"):
            if(self.debug): print(f"[Brain] detected follow-up question (short question): '{question}'")
            return True
            
        for indicator in followup_indicators:
            if f" {indicator} " in f" {question} " or question.startswith(indicator + " "):
                if(self.debug): print(f"[Brain] detected follow-up question with pattern: '{indicator}'")
                return True
                
        if len(question.split()) < 5:
            pronoun_indicators = ["it", "they", "them", "those", "these", "that", "this"]
            for pronoun in pronoun_indicators:
                if f" {pronoun} " in f" {question} ":
                    if(self.debug): print(f"[Brain] detected follow-up question (short with pronoun): '{question}'")
                    return True
        
        return False
        
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
        
    def _generate_opinion(self, question: str) -> str:
        """
        Generate a personalized response to opinion/preference questions.
        
        This function now acts as an interactive chatbot, generating follow-up questions and
        maintaining a natural conversation flow for opinion-based discussions.

        Args:
            question (str): The user's opinion or preference question.
        
        Returns:
            str: The generated response to the user's question.
        """
        if(self.debug): print(f"[Brain] generating opinion response for: '{question}'")
        
        conversation_context = []
        is_first_opinion_question = True
        
        if is_first_opinion_question:  
            opinion_count = 0
            for i in range(len(conversation_context)-1, -1, -1):
                # Skip the current question which is already marked as opinion
                if i == len(conversation_context)-1 and conversation_context[i]["role"] == "user":
                    continue
                if opinion_count >= 3:
                    break
                if conversation_context[i]["role"] != "user":
                    if "?" in conversation_context[i]["content"] or any(marker in conversation_context[i]["content"].lower() for marker in ["how about you", "what do you", "tell me more", "what about"]):
                        opinion_count += 1
            
            is_first_opinion_question = opinion_count == 0
            if(self.debug): print(f"[Brain] detected {'new' if is_first_opinion_question else 'ongoing'} opinion conversation with {opinion_count} previous turns")

        if not is_first_opinion_question and conversation_context:
            context_str = "\n\nCONVERSATION HISTORY:\n"
            for msg in conversation_context:
                role = "User" if msg["role"] == "user" else "You"
                context_str += f"{role}: {msg['content']}\n"
            prompt = context_str
        
        if is_first_opinion_question:
            prompt = textwrap.dedent(
                f"""    
                The user has asked: "{question}"

                Your response:
                """
            )
        else:
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
            if(self.debug): print(f"[Brain] error generating opinion: {e}")
            
            if is_first_opinion_question:
                return "Hey, that's a cool question! I'm still figuring out my thoughts on that. What kind of things do you like?"
            else:
                return "That's interesting! I'm still processing that one. What else is on your mind?"
        
    
    
    def _generate_answer(self, question: str, context_chunks: List[Dict], conversation_context: List[Dict] = None) -> str:
        
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
        for i, chunk in enumerate(context_chunks):
            chunk_text = [f"CONTEXT {i+1}:"]
            
            if 'description' in chunk and chunk['description']:
                chunk_text.append(f"Description: {chunk['description']}")
            
            if 'bullet_points' in chunk and isinstance(chunk.get('bullet_points'), list):
                points = []
                for point in chunk['bullet_points']:
                    if isinstance(point, str):
                        points.append(f"  • {point}")
                if points:
                    chunk_text.append("Key points:")
                    chunk_text.extend(points)
            
            for k, v in chunk.items():
                if k not in ('description', 'bullet_points') and isinstance(v, (str, int, float)):
                    chunk_text.append(f"{k.capitalize()}: {v}")
            
            formatted_context.append("\n".join(chunk_text))
        
        context_str = "\n\n".join(formatted_context)
        
        conversation_str = ""
        if conversation_context and len(conversation_context) > 0:
            if(self.debug): print(f"[Brain] including {len(conversation_context)} messages from conversation history")
            conversation_lines = []
            for msg in conversation_context:
                speaker = "User" if msg["role"] == "user" else "Assistant"
                conversation_lines.append(f"{speaker}: {msg['content']}")
            conversation_str = "\n".join(conversation_lines)
            if(self.debug): print(f"[Brain] conversation context length: {len(conversation_str)} chars")
        
        prompt = ""
        if conversation_str:
            prompt += textwrap.dedent(
                f"""
                CONVERSATION HISTORY:
                {conversation_str}
                """
            )
            
        prompt += textwrap.dedent(
            f"""
            KNOWLEDGE CONTEXT:
            {context_str}
            ANSWER:
            """ )

        if conversation_context and len(conversation_context) > 0:
            prompt += textwrap.dedent(
                """
                This appears to be a follow-up question. Use the conversation history to understand the context of the question.
                Make sure your answer maintains continuity with the previous conversation.
                """
            )
            
        prompt += textwrap.dedent(
            f"""

            USER QUESTION: {question}
            
            ANSWER:"""
        )
        
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
            chat_response = self.client.agents.complete(
                agent_id=agent_key,
                messages=[
                    {
                        "role": "user",
                        "content": f"{prompt}",
                    },
                ],
            )
            ans = chat_response.choices[0].message.content        
            final_answer = ans.strip()
            
            if(self.debug): print(f"[Brain] extracted final answer: {len(final_answer)} chars")
            return final_answer
        except Exception as e:
            if(self.debug): print(f"[Brain] error during answer generation: {e}")
            return "I encountered an error while trying to generate an answer based on the information I found. Please try asking your question again."

    def _generate_answer(self, question: str, context_string: str, conversation_context: List[Dict] = None) -> str:
        
        """
        LLM final step: fuse context → natural-language answer.
        
        Args:
            question (str): The user's question to answer.
            context_chunks (List[Dict]): The list of context chunks retrieved from memory.
            conversation_context (List[Dict], optional): The recent conversation context to include in the answer.

        Returns:
            str: The generated answer to the user's question.
        """

        
        if not context_string:
            
            if(self.debug): print("[Brain] no context string available for answer generation")
            
            return "Sorry dude. I don't have enough information to answer your question. I couldn't find relevant content in my memory or from web searches."

        
        if(self.debug): print(f"[Brain] generating answer using {context_string} context string")
        
        context_str = context_string
        conversation_str = ""
        if conversation_context and len(conversation_context) > 0:
            if(self.debug): print(f"[Brain] including {len(conversation_context)} messages from conversation history")
            conversation_lines = []
            for msg in conversation_context:
                speaker = "User" if msg["role"] == "user" else "Assistant"
                conversation_lines.append(f"{speaker}: {msg['content']}")
            conversation_str = "\n".join(conversation_lines)
            if(self.debug): print(f"[Brain] conversation context length: {len(conversation_str)} chars")
        
        prompt = ""
        if conversation_str:
            prompt += textwrap.dedent(
                f"""
                CONVERSATION HISTORY:
                {conversation_str}
                """
            )
            
        prompt += textwrap.dedent(
            f"""
            KNOWLEDGE CONTEXT:
            {context_str}
            ANSWER:
            """ )

        if conversation_context and len(conversation_context) > 0:
            prompt += textwrap.dedent(
                """
                This appears to be a follow-up question. Use the conversation history to understand the context of the question.
                Make sure your answer maintains continuity with the previous conversation.
                """
            )
            
        prompt += textwrap.dedent(
            f"""

            USER QUESTION: {question}
            
            ANSWER:"""
        )
        
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
            chat_response = self.client.agents.complete(
                agent_id=agent_key,
                messages=[
                    {
                        "role": "user",
                        "content": f"{prompt}",
                    },
                ],
            )
            ans = chat_response.choices[0].message.content        
            final_answer = ans.strip()
            
            if(self.debug): print(f"[Brain] extracted final answer: {len(final_answer)} chars")
            return final_answer
        except Exception as e:
            if(self.debug): print(f"[Brain] error during answer generation: {e}")
            return "I encountered an error while trying to generate an answer based on the information I found. Please try asking your question again."


    async def async_llm_call(self, prompt: str, temp: float, output_type: Optional[str] = None, json_schema: Optional[Dict[str, Any]] = None) -> str:
        """
        Helper function to call the LLM with a prompt and return the response.
        Args:
            prompt (str): The prompt to send to the LLM.
            temp (float): Temperature for the LLM response.
            output_type (str, optional): Output format type. Use "json_object" for JSON responses.
            json_schema (Dict[str, Any], optional): JSON schema to validate the response structure.
        Returns:
            str: The response from the LLM.
        """
        try:
            model = "ministral-8b-latest"
            chat_params = {
                "model": model,
                "temperature": temp,
                "messages": [UserMessage(content=prompt)],
            }
            
            if output_type:
                chat_params["response_format"] = {"type": output_type}
            
            if json_schema and output_type == "json_object":
                chat_params["response_format"]["schema"] = json_schema
            
            chat_response = await self.client.chat.complete_async(**chat_params)
            return chat_response.choices[0].message.content
            
        except Exception as e:
            if(self.debug): print(f"[Brain] LLM call failed: {e}")
            return "Error generating response"

    def sync_llm_call(self, prompt: str, temp: float, output_type: Optional[str] = None, json_schema: Optional[Dict[str, Any]] = None) -> str:
        """
        Synchronous LLM call using Mistral API directly.
        Args:
            prompt (str): The prompt to send to the LLM.
            temp (float): Temperature for the LLM response.
            output_type (str, optional): Output format type. Use "json_object" for JSON responses.
            json_schema (Dict[str, Any], optional): JSON schema to validate the response structure.
        Returns:
            str: The response from the LLM.
        """
        try:
            model = "open-mistral-nemo"
            chat_params = {
                "model": model,
                "temperature": temp,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            }
            
            if output_type:
                chat_params["response_format"] = {"type": output_type}

            if json_schema and output_type == "json_object":
                chat_params["response_format"]["json_schema"] = json_schema
            
            chat_response = self.client.chat.complete(**chat_params)
            return chat_response.choices[0].message.content
            
        except Exception as e:
            if(self.debug): print(f"[Brain] Sync LLM call failed: {e}")
            return "Error generating response"

    def get_relevant_context(self, query: str, max_results: int = 5, relevance_threshold: float = 0.2) -> List[Dict[str, Any]]:
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
            if not isinstance(node.data, dict) or not node.data:
                return 0.0
                
            data_text = node.data.get("description", "")
            title = node.topic if hasattr(node, 'topic') else ''
            all_text = f"{title} + {data_text}"
            all_text = all_text.lower()
            
            keyword_matches = 0
            total_keywords = len(keywords)
            
            for kw in keywords:
                if isinstance(kw, str) and kw.lower() in all_text:
                    keyword_matches += 1
                    # Boost score for title matches (more important)
                    if kw.lower() in title.lower():
                        keyword_matches += 0.5
            
            if total_keywords == 0:
                return 0.0
                
            base_score = keyword_matches / total_keywords
            
            content_boost = min(len(data_text) / 500.0, 1.0)  # Normalize content length
            
            final_score = base_score + (content_boost * 0.2)
            final_score = min(final_score, 1.0)  # Cap at 1.0
            
            return final_score

        def dfs_search(node: 'SupabaseTopicNode', visited: set, scored_nodes: list, high_scored_nodes: list) -> None:
            """
            DFS traversal that only explores relevant branches.

            Args:
                node: The current SupabaseTopicNode to explore.
                visited: Set of already visited node IDs to avoid cycles.
                scored_nodes: List to collect scored nodes for final sorting.

            Returns:
                None: This function modifies scored_nodes in place.
            """
            if len(scored_nodes) > max_results:
                if(self.debug): print(f"[Brain] Reached max results limit of {max_results}, stopping DFS")
                return
            if node.node_id in visited:
                return
            visited.add(node.node_id)
            
            if isinstance(node.data, dict) and node.data:
                score = calculate_relevance_score(node)
                if score > relevance_threshold:
                    scored_nodes.append((score, node.data, node.topic))
                    if(self.debug): print(f"[Brain] Found relevant: '{node.topic}' (score: {score:.3f})")
                    if score > relevance_threshold:
                        high_scored_nodes.append(node)
            
            # Continue DFS on all children since this branch is relevant
            for child in node.children:
                if child.node_id not in visited:
                    dfs_search(child, visited, scored_nodes, high_scored_nodes)
        if(self.debug):
            print("getting keywords from llm")
        keywords = self._get_likely_keywords_from_llm(query)

        if not keywords:
            if(self.debug): print(f"[Brain] No keywords generated from LLM, falling back to basic keyword extraction for query: '{query}'")
            keywords = set(re.findall(r'\w+', query.lower()))
        if(self.debug): 
            print(f"[Brain] Searching for keywords: {keywords}")


        visited = set()
        scored_nodes = []
        high_scored_nodes = []
        if(self.debug): print(f"[Brain] Starting DFS search from root: '{self.memory.topic}'")
        dfs_search(self.memory, visited, scored_nodes, high_scored_nodes)

        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        
        # Return just the data from top results
        results = [item[1] for item in scored_nodes[:max_results]]
        
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
            Return ONLY a JSON Array of 30 possible categories that this query could fit into. Make sure to include 'Politics' for any query that references a place. Include 'Current Events' for any question that is asking about something happening now or recently. Make sure to include country names or any other very broad topics that could be related to the query too.
            Query: {query}
            """
        )
        json_schema={
            "type": "array",
            "items": {"type": "string"}
        }

        output = self.sync_llm_call(prompt, temp=0.2, output_type='json_object', json_schema=json_schema)
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
            queue.extend([(t, 0) for t in random.sample(default_roots, k=min(breadth, len(default_roots)))])
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
            try:
                self.answer(question, max_search_results=max_search_results) 
                explored.append(question)
            except Exception as e:
                if(self.debug): print(f"[Brain] ⚠️ exploration error: {e}")

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

        if(self.debug): print(f"[Brain] exploration complete - {len(explored)} topics added to memory")
        return explored
    
    async def process_search_results_parallel(self, search_hits, user_question: str) -> List[str]:
        """
        Process search results in parallel - scraping and summarizing concurrently
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
            
            raw = await asyncio.get_event_loop().run_in_executor(None, self._scrape, hit["url"])
            
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
        
        if(self.debug): print("[Brain] sending multi-website summarization prompt")
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
            if(self.debug): print(f"[Brain] multi-website summary response {response}")
            context.append(response)
            # Parse the JSON response
            structured_results = json.loads(response)
            
            # Store each summary
            async def store_single_result(summary: Dict) -> bool:
                """Store a single processed result"""
                try:
                    import concurrent.futures
                    # Generate topic path using a thread pool executor
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        path = await asyncio.get_running_loop().run_in_executor(
                            executor, self._generate_topic_path, user_question, summary.get("main_idea", "Unknown"), summary
                        )
                    
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
            
            if(self.debug): print(f"[Brain] storing {len(structured_results)} summary results")

            if structured_results:
                store_tasks = [store_single_result(item) for item in structured_results]
                store_results = await asyncio.gather(*store_tasks, return_exceptions=True)
                
                successful_stores = sum(1 for result in store_results if result is True)
                if(self.debug): print(f"[Brain] stored information from {successful_stores}/{len(structured_results)} search results")
            return context
            
        except Exception as e:
            if(self.debug): print("[Brain] failed to search web: ", e)
            return "No Context"

if __name__ == "__main__":
    brain = Brain()
    brain.explore_topics_autonomously(base_topics=["Artificial Intelligence", "Quantum Computing"], max_depth=2, breadth=3, max_total_topics=10, subtopics_per_topic=2, delay=0.5, max_search_results=3)