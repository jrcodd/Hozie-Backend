#I'm running into rate limit issues when using the new async plan. Have to look into using my credits which are not being used for some reason. Have to figure out how to switch to the paid plan

import json
import os
import re
import textwrap
import time
import random
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Sequence, Tuple, Optional

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS 
from mistralai import Mistral

from supabase_topic_node import SupabaseTopicNode 
from supabase_chat_history import SupabaseChatHistory 

def  _get_time() -> str:
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

    def __init__(self) -> None:
        """
        Connect to Mistral API and create or load memory tree.
        
        Args:
            api_key: Mistral API key. If not provided, it will be read from the environment variable MISTRAL_API_KEY.
        """
        self.api_key = os.environ.get('MISTRAL_API_KEY')
        self._init_mistral()
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
                        print(f"[Brain] Added result: {title[:50]}...")
                
                return results
            else:
                print(f"[Brain] Brave API error: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            print(f"[Brain] Brave search error: {e}")
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
        
        print(f"[Brain] scraping: {url}")
        
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
                print(f"[Brain] retrying with different user agent")   

            try:
                response = requests.get(url, timeout=10, headers=headers, allow_redirects=True)
                response.raise_for_status()
                
                content_type = response.headers.get('Content-Type', '').lower()
                if 'text/html' not in content_type:
                    print(f"[Brain] skipping non-HTML content: {content_type}")
                    return None
                    
                return response.text
            except requests.exceptions.HTTPError as e:
                #Only retry once to prevent an infinite loop
                if not retry and hasattr(e, 'response') and e.response.status_code in [403, 429]:
                    return fetch_url(retry=True)
                print(f"[Brain] HTTP error: {e}")
                return None
            except Exception as e:
                print(f"[Brain] request error: {e}")
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
            print(f"[Brain] error parsing content: {e}")
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
            print(f"[Brain] text too short: ({len(raw_text)} chars)")
            return {}

        print(f"[Brain] attempting to summarize {len(raw_text)} chars of text")
        
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
            print("[Brain] generating summary with LLM...")
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
            print(f"[Brain] summarized output length: {len(response)} chars")
            
            try:
                data = json.loads(response)
                print(f"[Brain] successfully extracted JSON with keys: {list(data.keys())}")
                
                required_keys = ["main_idea", "description", "bullet_points", "source"]
                for key in required_keys:
                    if key not in data:
                        data[key] = "" if key != "bullet_points" else []
                
                if not isinstance(data["bullet_points"], list):
                    print("[Brain] bullet_points is not formatted correctly, trying to convert to list")
                    data["bullet_points"] = [data["bullet_points"]] if data["bullet_points"] else []
            
                return data
                
            except json.JSONDecodeError as e:
                print(f"[Brain] JSON parsing error: {e}")

        except Exception as e:
            print(f"[Brain] Error during summarization: {e}")
            if not output:
                print("[Brain] No output generated, returning empty node")
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
                print(f"[Brain] found case-insensitive match: {child.topic} for {next_topic}")
                return self._check_similar_path(child, target_path[1:], current_path + [child.topic])
                
            # 2. Check if one is a prefix/suffix of the other
            # This will match "Black Holes" with "Black Hole" or "Black Holes Formation"
            if child.topic.lower().startswith(next_topic.lower()) or next_topic.lower().startswith(child.topic.lower()):
                print(f"[Brain] found prefix/suffix match: {child.topic} for {next_topic}")
                return self._check_similar_path(child, target_path[1:], current_path + [child.topic])
                
            # 3. Check for word-level similarity
            child_words = set(re.findall(r'\w+', child.topic.lower()))
            target_words = set(re.findall(r'\w+', next_topic.lower()))
            
            # If there's significant overlap, consider it similar
            if child_words and target_words:
                overlap = child_words.intersection(target_words)
                similarity = len(overlap) / max(len(child_words), len(target_words))
                
                if similarity > 0.8:  
                    print(f"[Brain] found similar node: {child.topic} for {next_topic} (similarity: {similarity:.2f})")
                    return self._check_similar_path(child, target_path[1:], current_path + [child.topic])
        
        # 4. Special case handling for common hierarchical relationships
        science_categories = ["physics", "astronomy", "chemistry", "biology", "climetology", "geology", "earth science", "environmental science"]
        if (node.topic.lower() == "science" and next_topic.lower() in science_categories) or \
           (next_topic.lower() == "science" and node.topic.lower() in science_categories):
            # If this is a science-related path, check if we're duplicating the hierarchy
            print(f"[Brain] detected potential science category duplication: {node.topic} vs {next_topic}")
            
            # For "Science/Physics" and "Physics" patterns, prefer the more specific
            if node.topic.lower() == "science" and next_topic.lower() in science_categories:
                for child in node.children:
                    if child.topic.lower() == next_topic.lower():
                        print(f"[Brain] found science subcategory: {child.topic}")
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
            print("[Brain] ERROR: Empty topic path provided, using fallback path")
            topic_path = ["Uncategorized", "WebContent"]
            
        if not data:
            print("[Brain] ERROR: Empty data provided, skipping memory insertion")
            return
            
        print(f"[Brain] inserting data into memory at path: {'/'.join(topic_path)}")
        
        try:
            if topic_path[0] == self.memory.topic:
                found_similar, existing_node, existing_path = self._check_similar_path(self.memory, topic_path[1:])
            else:
                found_similar, existing_node, existing_path = self._check_similar_path(self.memory, topic_path)
            
            if found_similar:
                print(f"[Brain] found similar existing path: {'/'.join(existing_path)}")
                current_node = existing_node
            else:
                print(f"[Brain] creating new path: {'/'.join(topic_path)}")
                
                current_node = self.memory
                for i, topic in enumerate(topic_path):
                    child = current_node.find_child(topic)
                    if not child:
                        if i == len(topic_path) - 1:
                            child = current_node.add_child(topic, data)
                            print(f"[Brain] created leaf node: {child.topic}")
                        else:
                            child = current_node.add_child(topic)
                            print(f"[Brain] created intermediate node: {child.topic}")
                    current_node = child
            
            node = current_node
            print(f"[Brain] node created/found: {node.topic}")
                        
            # Merge dictionaries newer values overwrite older ones
            if isinstance(node.data, dict) and node.data:
                print(f"[Brain] merging with existing data ({len(node.data)} keys)")
                node.data.update(data)
            else:
                print("[Brain] setting new data")
                node.data = data

            meta_entry = {
                "url": source_url,
                "timestamp":  _get_time(),
                "confidence": confidence,
            }
            node.metadata.setdefault("sources", []).append(meta_entry)
            node.metadata["last_updated"] =  _get_time()
            print(f"[Brain] successfully stored data under: {'/'.join(topic_path)}")
                        
        except Exception as e:
            print(f"[Brain] ERROR during memory insertion: {e}")

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
        
        print("[Brain] generating proper topic path...")
        """
        0.2 temp mistral small prompt:
        Create a logical knowledge hierarchy that:
            1. Starts with the broadest domain (Science, History, etc.)
            2. Follows with subdomain (e.g., Physics, Ancient History)
            3. Continues with topic area (e.g., Quantum Mechanics, Roman Empire)
            4. Ends with the specific concept (e.g., Wave Function, Julius Caesar)

        Return ONLY a JSON array of path elements from general to specific, such as: ["Science", "Astronomy", "Celestial Objects", "Black Holes"]
        Do not include explanations, just the array.
        Analyze the following information and generate a hierarchical knowledge path for storing in a knowledge tree. The path should follow academic taxonomy conventions starting with broad domains and narrowing to specifics Keep it short.
        """
        agent_key = "ag:a2eb8171:20250526:hozie-generate-content-path:c4db64e4"
        chat_response = self.client.agents.complete(
            agent_id=agent_key,
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt}",
                },
            ],
        )
        output = chat_response.choices[0].message.content
                
        match = re.search(r'\[.*\]', output, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                path = json.loads(json_str)
                if isinstance(path, list) and all(isinstance(item, str) for item in path) and len(path) >= 2:
                    print(f"[Brain] generated structured path: {path}")
                    return path
            except json.JSONDecodeError:
                pass
        else:
            print("[Brain] no valid JSON path found in output")
            path = ["Uncategorized", "WebContent"]

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
        print(f"[Brain] retrieving context for query: '{query}'")
        try:
            node_count = len([self.memory] + self.memory.get_all_descendants())
            
            if node_count < k:
                print("[Brain] memory tree is empty or nearly empty using web search")
                return []
            
            #TODO: remove the depth limit
            contexts = self.memory.get_relevant_context(query, depth=500, max_results=k*3)

            print(f"[Brain] found {len(contexts)} context chunks")

            if contexts:
                unique_contexts = []
                description_set = set()   

                # skip duplicates based on description fingerprint to reduce prompt size
                for ctx in contexts:
                    if ctx.get('description') and ctx.get('bullet_points'):
                        desc = ctx.get('description', '')
                        if not desc or len(desc) <= 30:
                            unique_contexts.append(ctx)
                            continue

                        fingerprint = desc[:40].strip().lower()
                        if fingerprint in description_set:
                            print(f"[Brain] skipping duplicate context: {fingerprint}")
                            continue
                        
                        description_set.add(fingerprint)
                        unique_contexts.append(ctx)
                                         
                # Limit to top k unique results
                results = unique_contexts[:k]
                print(f"[Brain] returning top {len(results)} unique context chunks")
                return results
            else:
                print("[Brain] no relevant context found")
                #dont call web search here bc it could cause an infinite loop if something is broken
                return []
                
        except Exception as e:
            print(f"[Brain] error during context retrieval: {e}")
            return []

    def _print_tree(self, node: SupabaseTopicNode, indent: str = " ") -> None:
        """
        Pretty-print the SupabaseTopicNode hierarchy.
        
        Args:
            node: The current node in the tree to print.
            indent: The indentation string for pretty-printing.
        """
        print(f"{indent}- {node.topic}")
        for child in node.children:
            Brain._print_tree(child, indent + indent)

    def _is_context_relevant(self, query: str, context: List[Dict]) -> bool:
        """
        Check if the retrieved context is actually relevant to the query.
        This method analyzes the context against the user's query to determine relevance. no llm calls are made here, just regex and string matching.

        Args:
            query: The user's search query or question.
            context: The list of context chunks retrieved from memory.

        Returns:
            bool: True if the context is relevant to the query, False otherwise.
        """
        if context:
            print("[Brain] checking context relevance...")
            
            # Extract main topic words from query (lowercase words with at least 3 chars) some exceptions could be LA or US or UK, but usually the memory tree will have these spelled out as "Los Angeles", "United States", "United Kingdom" etc.
            main_topic_words = set(re.findall(r'\b[a-z]{3,}\b', query.lower()))
            
            # Also extract important 2-word phrases for more accurate matching
            main_phrases = re.findall(r'\b([a-z]{3,}\s+[a-z]{3,})\b', query.lower())
            
            if len(main_topic_words) <= 2:
                # For very short queries accept any context that has at least one matching word

                print(f"[Brain] short query detected with only {len(main_topic_words)} key words")

                for item in context:
                    if not isinstance(item, dict):
                        continue

                    context_text = ""
                    if "description" in item:
                        context_text += " " + item["description"].lower()
                        
                    if "bullet_points" in item and isinstance(item["bullet_points"], list):
                        for point in item["bullet_points"]:
                            if isinstance(point, str):
                                context_text += " " + point.lower()
                    
                    # For short queries, if any important word appears, consider it relevant
                    for word in main_topic_words:
                        if word in context_text and len(word) > 3: 
                            return True
                            
                    for phrase in main_phrases:
                        if phrase in context_text:
                            print(f"[Brain] short query matched exact phrase: {phrase}")
                            return True
                
                return False
            
            # Regular relevance checking for normal-length queries
            context_relevance_scores = []
            for item in context:
                if isinstance(item, dict):                    
                    context_text = ""
                    if "description" in item:
                        context_text += " " + item["description"].lower()
                        
                    if "bullet_points" in item and isinstance(item["bullet_points"], list):
                        for point in item["bullet_points"]:
                            if isinstance(point, str):
                                context_text += " " + point.lower()

                    context_words = set(re.findall(r'\b[a-z]{3,}\b', context_text))
                    if main_topic_words and context_words:
                        overlap = main_topic_words.intersection(context_words)
                        # Weight by importance - longer words are usually more significant
                        weighted_score = sum(len(word) for word in overlap) / sum(len(word) for word in main_topic_words)
                        basic_score = len(overlap) / len(main_topic_words)
                        # Combine scores with weighting toward important words
                        score = (weighted_score + basic_score) / 2
                        context_relevance_scores.append(score)
                        
                        # Strongly boost score if we have exact phrase matches which are more meaningful
                        for phrase in main_phrases:
                            if phrase in context_text:
                                print(f"[Brain] matched exact phrase: {phrase}")
                                context_relevance_scores[-1] += 0.3
                
                if context_relevance_scores:
                    avg_relevance = sum(context_relevance_scores) / len(context_relevance_scores)
                    # Cap at 1.0 for cleaner reporting
                    avg_relevance = min(avg_relevance, 1.0)
                    print(f"[Brain] context relevance score: {avg_relevance:.2f}")
                    
                    #TODO: test this threshold, right now it is low to keep the system responsive
                    is_relevant = avg_relevance > 0.07  # 7% overlap of query words with context
                    
                    if is_relevant:
                        print("[Brain] context is relevant to the query")
                    else:
                        print("[Brain] context is NOT relevant to the query, need to search")
                        
                    return is_relevant
                
        return False

    def answer(self, user_question: str, max_search_results: int = 5) -> str:
        """
        Main entry-point: answer a user question using RAG.
        Args:
            user_question (str): The user's question to answer
            max_search_results (int, optional): Maximum number of websites to search when looking for information. Defaults to 5.
        Returns:
            str: The generated answer to the user's question.
        """
        print(f"[Brain] processing question: '{user_question}'")
        
        # Check if this is a personal opinion/preference question and skip web searching for these
        is_opinion_question = self._is_opinion_question(user_question)
        if is_opinion_question:
            print("[Brain] detected opinion/preference question, generating personalized response")
            response = self._generate_opinion(user_question)
            return response
        
        #TODO: add previous conversation to the context if this is a follow-up question
        is_followup = self._is_followup_question(user_question)
        if is_followup:
            # access supabase_chat_history to get the recent context
            print("[Brain] detected follow up question, retrieving recent context")
            recent_context = []
        
        print("[Brain] checking internal memory...")
        context = self._retrieve_context(user_question, k=5)
        
        if context and self._is_context_relevant(user_question, context):
            print(f"[Brain] found {len(context)} relevant contexts in memory")
            print("[Brain] internal context sufficient.")
            
            #debug print context
            for i, ctx in enumerate(context):
                desc = ctx.get('description', '')[:100] + '...' if ctx.get('description') else 'No description'
                points = ctx.get('bullet_points', [])
                point_preview = points[0][:50] + '...' if points and isinstance(points[0], str) and len(points[0]) > 50 else points[0] if points else 'No bullet points'
                print(f"[Brain] Answer Context {i+1}: {desc}")
                print(f"[Brain] Answer Context {i+1} Point: {point_preview}")
            
            recent_context = recent_context if is_followup else []
            response = self._generate_answer(user_question, context, recent_context)
            return response
        else:
            if context:
                print("[Brain] found some context but it's not sufficient, proceeding to web search")
            else:
                print("[Brain] no relevant context found in memory, proceeding to web search")
            
            search_q = self._rewrite_query(user_question)
            print(f"[Brain] search query rewritten to: '{search_q}'")
            
            search_hits = self._search(search_q, max_results=max_search_results)
            print(f"[Brain] found {len(search_hits)} search results")
            
            # Capture the return value from the async function
            response = asyncio.run(self.process_search_results_parallel(search_hits, user_question))
            return response
    # ------------------------------------------------------------------

    def _is_followup_question(self, question: str) -> bool:
        """Determine if the question is a follow-up to a previous conversation."""
        # Always return False for shared brain - no conversation history
        return False
            
        # Look for indicators of follow-up questions
        followup_indicators = [
            # Pronouns that refer to something previously mentioned
            "it", "they", "them", "those", "these", "that", "this",
            
            # Direct references to previous information
            "as mentioned", "as you said", "as we discussed", "earlier", "previous",
            "you just said", "you told me", "you mentioned",
            
            # Questions that build on previous context
            "and what about", "what else", "tell me more", "can you elaborate",
            "why is that", "how does that work", "could you explain", 
            "what if", "is there more", "go on", "continue", "furthermore",
            
            # Short standalone questions that rely on context
            "why","who", "what if", "which of those", 
            
            # Direct continuations
            "also", "additionally", "moreover", "and", "but"
        ]
        
        # Very short questions are often follow-ups
        if len(question.split()) <= 3 and not question.startswith("what is") and not question.startswith("how to"):
            print(f"[Brain] detected follow-up question (short question): '{question}'")
            return True
            
        # Check for follow-up indicators
        for indicator in followup_indicators:
            if f" {indicator} " in f" {question} " or question.startswith(indicator + " "):
                print(f"[Brain] detected follow-up question with pattern: '{indicator}'")
                return True
                
        # Check for questions that make little sense without context
        # These are often short questions with pronouns
        if len(question.split()) < 5:
            pronoun_indicators = ["it", "they", "them", "those", "these", "that", "this"]
            for pronoun in pronoun_indicators:
                if f" {pronoun} " in f" {question} ":
                    print(f"[Brain] detected follow-up question (short with pronoun): '{question}'")
                    return True
        
        return False
        
    def _is_opinion_question(self, question: str) -> bool:
        """Determine if the question is asking for opinions or preferences."""
        question = question.lower()
        
        # Detect common opinion/preference question patterns
        opinion_indicators = [
            # Direct questions about preferences
            
            "what do you like about", "how are you feeling", "what do you like", 
            "what do you enjoy", "what do you prefer", 
            "what kind of", "which do you prefer", "which do you like",
            "do you like", "do you enjoy", "do you prefer",
            "what's your opinion", "what is your opinion", "what are your thoughts on",

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

        
        # Check if any opinion indicators are present
        for indicator in opinion_indicators:
            question = question.replace("?", "").replace("'", "")
            if (indicator + " ") in (question + " "):
                print(f"[Brain] detected opinion question with pattern: '{indicator}'")
                return True
                
        return False
        
    def _generate_opinion(self, question: str) -> str:
        """Generate a personalized response to opinion/preference questions.
        
        This function now acts as an interactive chatbot, generating follow-up questions and
        maintaining a natural conversation flow for opinion-based discussions.
        """
        print(f"[Brain] generating opinion response for: '{question}'")
        
        # For shared brain, treat all opinion questions as first-time
        conversation_context = []
        is_first_opinion_question = True
        
        # Skip conversation context for shared brain
        if False:  # Disabled for shared brain
            # Count recent turns to determine if we're in a conversation
            opinion_count = 0
            for i in range(len(conversation_context)-1, -1, -1):
                # Skip the current question which is already marked as opinion
                if i == len(conversation_context)-1 and conversation_context[i]["role"] == "user":
                    continue
                # If we've gone back 3 turns (or hit beginning), stop counting
                if opinion_count >= 3:
                    break
                # Check if assistant response includes chatbot-like elements (questions, conversation markers)
                if conversation_context[i]["role"] == "assistant":
                    if "?" in conversation_context[i]["content"] or any(marker in conversation_context[i]["content"].lower() for marker in ["how about you", "what do you", "tell me more", "what about"]):
                        opinion_count += 1
            
            is_first_opinion_question = opinion_count == 0
            print(f"[Brain] detected {'new' if is_first_opinion_question else 'ongoing'} opinion conversation with {opinion_count} previous turns")

        # Add conversation context if this is a follow-up
        if not is_first_opinion_question and conversation_context:
            context_str = "\n\nCONVERSATION HISTORY:\n"
            for msg in conversation_context:
                role = "User" if msg["role"] == "user" else "You"
                context_str += f"{role}: {msg['content']}\n"
            prompt = context_str
        
        # Add specific instructions based on whether this is a new conversation or a follow-up
        if is_first_opinion_question:
            prompt = textwrap.dedent(
                f"""    
                The user has asked: "{question}"

                Your response:
                """
            )
        else:
            # For follow-up exchanges, focus more on natural conversation flow
            prompt += textwrap.dedent(
                f"""
                The user's latest message is: "{question}"
                
                Your response:
                """
            )

        try:
            # Generate the response with appropriate temperature for personality
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
            
            # Debug output
            print(f"[Brain] opinion prompt length: {len(prompt)} chars")
            print(f"[Brain] opinion response length: {len(response)} chars")
            
            # Extract just the response portion - remove any added quotation marks
            if "Your response:" in response:
                response = response.split("Your response:")[-1].strip()
                
            # Remove any surrounding quotes if present
            response = response.strip('"')
            
            print(f"[Brain] generated opinion response: {len(response)} chars")
            
            return response
            
        except Exception as e:
            print(f"[Brain] error generating opinion: {e}")
            
            # Fallback response if something goes wrong
            if is_first_opinion_question:
                return "Hey, that's a cool question! I'm still figuring out my thoughts on that. What kind of things do you like?"
            else:
                return "That's interesting! I'm still processing that one. What else is on your mind?"
        
    
    
    def _generate_answer(self, question: str, context_chunks: List[Dict], conversation_context: List[Dict] = None) -> str:
        
        """LLM final step: fuse context → natural-language answer."""

        
        if not context_chunks:
            
            print("[Brain] no context chunks available for answer generation")
            
            return "I don't have enough information to answer your question. I couldn't find relevant content in my memory or from web searches."

        
        print(f"[Brain] generating answer using {len(context_chunks)} context chunks")
        
        # Create short summaries of context chunks for logging
        def format_value_for_log(value):
            if isinstance(value, str) and len(value) > 50:
                return value[:50] + "..."
            elif isinstance(value, list):
                return f"[{len(value)} items]"
            else:
                return value
                
        log_context = [
            {k: format_value_for_log(v) for k, v in chunk.items()}
            for chunk in context_chunks
        ]
        print(f"[Brain] context summary: {json.dumps(log_context)}")
        
        
        # Format context for LLM prompt
        formatted_context = []
        for i, chunk in enumerate(context_chunks):
            # Start with the context number
            chunk_text = [f"CONTEXT {i+1}:"]
            
            # Add description if available
            if 'description' in chunk and chunk['description']:
                chunk_text.append(f"Description: {chunk['description']}")
            
            # Add bullet points if available
            if 'bullet_points' in chunk and isinstance(chunk.get('bullet_points'), list):
                points = []
                for point in chunk['bullet_points']:
                    if isinstance(point, str) and len(point) > 5:  # Skip very short points
                        points.append(f"  • {point}")
                if points:
                    chunk_text.append("Key points:")
                    chunk_text.extend(points)
            
            # Add other relevant fields
            for k, v in chunk.items():
                if k not in ('description', 'bullet_points') and isinstance(v, (str, int, float)):
                    chunk_text.append(f"{k.capitalize()}: {v}")
            
            # Join all parts with newlines
            formatted_context.append("\n".join(chunk_text))
        
        # Join all contexts with double newlines
        context_str = "\n\n".join(formatted_context)
        
        # Add conversation context if available
        conversation_str = ""
        if conversation_context and len(conversation_context) > 0:
            print(f"[Brain] including {len(conversation_context)} messages from conversation history")
            conversation_lines = []
            for msg in conversation_context:
                # Format chat history as a conversation
                speaker = "User" if msg["role"] == "user" else "Assistant"
                conversation_lines.append(f"{speaker}: {msg['content']}")
            conversation_str = "\n".join(conversation_lines)
            print(f"[Brain] conversation context length: {len(conversation_str)} chars")
        
        prompt = ""
        # Add conversation history context if available
        if conversation_str:
            prompt += textwrap.dedent(
                f"""
                CONVERSATION HISTORY:
                {conversation_str}
                """
            )
            
        # Continue with knowledge context and instructions
        prompt += textwrap.dedent(
            f"""
            KNOWLEDGE CONTEXT:
            {context_str}
            ANSWER:
            """ )

        # For follow-up questions, add specific instructions
        if conversation_context and len(conversation_context) > 0:
            prompt += textwrap.dedent(
                """
                This appears to be a follow-up question. Use the conversation history to understand the context of the question.
                Make sure your answer maintains continuity with the previous conversation.
                """
            )
            
        # Complete the prompt
        prompt += textwrap.dedent(
            f"""

            USER QUESTION: {question}
            
            ANSWER:"""
        )
        
        print("[Brain] sending prompt to LLM for final answer generation")
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

            print(f"[Brain] raw answer from LLM: {len(ans.strip())} chars")
            
            # Debug output for troubleshooting
            print(f"[Brain] final prompt length: {len(prompt)} chars")
            print(f"[Brain] raw answer length: {len(ans)} chars")
            
            # Extract only the answer portion
            answer_marker = "ANSWER:"
            if answer_marker in ans:
                # Use the text after the last occurrence of ANSWER:
                final_answer = ans.split(answer_marker)[-1].strip()
            else:
                # Try other possible markers
                markers = ["USER QUESTION:", "Answer:", "Response:"]
                for marker in markers:
                    if marker in ans:
                        parts = ans.split(marker)
                        if len(parts) > 1:
                            # Take the last part after this marker
                            final_answer = parts[-1].strip()
                            break
                else:
                    # If no markers found, use the whole response
                    final_answer = ans.strip()
            
            print(f"[Brain] extracted final answer: {len(final_answer)} chars")
            return final_answer
        except Exception as e:
            print(f"[Brain] error during answer generation: {e}")
            return "I encountered an error while trying to generate an answer based on the information I found. Please try asking your question again."


    # =====================================================================
    #                       💾  MEMORY PERSISTENCE
    # =====================================================================
    
    def _load_memory(self) -> bool:
        """Load the memory tree from a JSON file."""
        if not self.memory_file or not os.path.exists(self.memory_file):
            # No memory file to load
            return False
            
        try:
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                memory_dict = json.load(f)
                
            # Remove metadata before loading
            if "_meta" in memory_dict:
                save_time = memory_dict["_meta"].get("saved_at", "unknown")
                version = memory_dict["_meta"].get("version", "unknown")
                print(f"[Brain] loading memory saved at {save_time} (version {version})")
                del memory_dict["_meta"]
                
            # Create the memory tree from the dictionary
            self.memory = SupabaseTopicNode.from_dict(memory_dict)
            print(f"[Brain] successfully loaded memory from {self.memory_file}")
            return True
            
        except Exception as e:
            print(f"[Brain] ERROR loading memory from {self.memory_file}: {e}")
            # Start with a fresh memory tree
            self.memory = SupabaseTopicNode("Knowledge Base")
            return False

# -----------------------------------------------------------------------------
# Example usage (manual run)
# -----------------------------------------------------------------------------
    def explore_topics_autonomously(self,
                                      base_topics: Optional[List[str]] = None,
                                      *,
                                      max_depth: int = 3,
                                      breadth: int = 5,
                                      max_total_topics: int = 50,
                                      subtopics_per_topic: int = 5,
                                      delay: float = 0.5,
                                      temperature: float = 0.7,
                                      max_search_results: int = 5) -> List[str]:
        """💡 **Enhanced autonomous exploration**

        This upgraded routine aggressively expands the knowledge tree by performing a
        *breadth-first* crawl over a topic hierarchy that it generates on-the-fly with
        the LLM.  It repeatedly asks the model for *sub-topics* of each queued topic
        and dives up to *max_depth* levels deep, respecting *max_total_topics* so we
        don't explode our token quota.

        Parameters
        ----------
        base_topics : list[str] | None
            Seed topics to start from.  If *None*, we seed with a handful of very
            broad academic domains.
        max_depth : int, default 3
            How many hierarchical levels to descend (root = 0).
        breadth : int, default 5
            Maximum number of topics to fetch *per level*.
        max_total_topics : int, default 50
            Hard ceiling on how many individual topics we explore in a single run.
        subtopics_per_topic : int, default 5
            How many sub-topics to request from the LLM for each topic.
        delay : float, default 0.5
            Pause (seconds) between web calls so we don't hammer rate-limits.
        temperature : float, default 0.7
            Sampling temperature for LLM when brainstorming sub-topics.
        max_search_results : int, default 5
            Maximum number of websites to scrape for each topic.

        Returns
        -------
        list[str]
            A flat list of *every* topic question that was pushed through
            ``self.answer()`` during the session - useful for logs/stats.
        """

        print(f"[Brain] 🚀 starting enhanced autonomous exploration - target {max_total_topics} topics ...")

        explored: List[str] = []               # flat list of explored questions
        queue: List[Tuple[str, int]] = []      # (topic, depth)

        # ------------------------------------------------------------------
        # 1.  Seed queue
        # ------------------------------------------------------------------
        if base_topics:
            queue.extend([(t, 0) for t in base_topics[:breadth]])
            print(f"[Brain] seeding with user-supplied base topics: {base_topics[:breadth]}")
        else:
            default_roots = [
                "Science", "Technology", "Mathematics", "History", "Art",
                "Economics", "Psychology", "Biology", "Physics", "Computer Science"
            ]
            queue.extend([(t, 0) for t in random.sample(default_roots, k=min(breadth, len(default_roots)))])
            print(f"[Brain] seeding with default academic domains: {[q[0] for q in queue]}")

        # ------------------------------------------------------------------
        # 2.  Helper - expand a topic into subtopics using the LLM
        # ------------------------------------------------------------------
        def _generate_subtopics(topic: str, k: int) -> List[str]:
            prompt = textwrap.dedent(f"""
                List {k} highly specific sub-topics that a university-level
                researcher would study **within** the domain of "{topic}".
                • respond with one sub-topic per line, phrased as a direct
                  question beginning with *How*, *Why* or *What* so it can be
                  fed straight into a search engine.
            """)
            try:
                raw = self.gen(prompt, max_new_tokens=64, temperature=temperature)[0]["generated_text"]
                lines = [re.sub(r"^[•\-\d\.\s]*", "", l).strip() for l in raw.splitlines()]
                # keep unique, non-empty, question-like lines
                sub = []
                for l in lines:
                    if l and l[-1] == '?' and l not in sub:
                        sub.append(l)
                    if len(sub) >= k:
                        break
                return sub
            except Exception as e:
                print(f"[Brain] sub-topic generation failed for '{topic}': {e}")
                return []

        # ------------------------------------------------------------------
        # 3.  BFS over topic hierarchy
        # ------------------------------------------------------------------
        while queue and len(explored) < max_total_topics:
            topic, d = queue.pop(0)
            question = f"Tell me about {topic}" if not topic.endswith('?') else topic

            print(f"[Brain] ▸ exploring (depth {d}) - {question}")
            try:
                self.answer(question, max_search_results=max_search_results) 
                explored.append(question)
                self._save_memory()
            except Exception as e:
                print(f"[Brain] ⚠️ exploration error: {e}")

            # stop descending if depth limit reached or global budget exhausted
            if d >= max_depth or len(explored) >= max_total_topics:
                continue

            # brainstorm sub-topics and enqueue them
            children = _generate_subtopics(topic, subtopics_per_topic)
            random.shuffle(children)            # add variety
            for child in children[:breadth]:
                if len(explored) + len(queue) >= max_total_topics:
                    break
                queue.append((child, d + 1))

            time.sleep(delay)

        print(f"[Brain] 🏁 exploration complete - {len(explored)} topics added to memory")
        return explored

    def _is_followup_question(self, question: str) -> bool:
        """Determine if the question is a follow-up to a previous conversation."""
        # Always return False for shared brain - no conversation history
        return False
            
        # Look for indicators of follow-up questions
        followup_indicators = [
            # Pronouns that refer to something previously mentioned
            "it", "they", "them", "those", "these", "that", "this",
            
            # Direct references to previous information
            "as mentioned", "as you said", "as we discussed", "earlier", "previous",
            "you just said", "you told me", "you mentioned",
            
            # Questions that build on previous context
            "and what about", "what else", "tell me more", "can you elaborate",
            "why is that", "how does that work", "could you explain", 
            "what if", "is there more", "go on", "continue", "furthermore",
            
            # Short standalone questions that rely on context
            "why","who", "what if", "which of those", 
            
            # Direct continuations
            "also", "additionally", "moreover", "and", "but"
        ]
        
        # Very short questions are often follow-ups
        if len(question.split()) <= 3 and not question.startswith("what is") and not question.startswith("how to"):
            print(f"[Brain] detected follow-up question (short question): '{question}'")
            return True
            
        # Check for follow-up indicators
        for indicator in followup_indicators:
            if f" {indicator} " in f" {question} " or question.startswith(indicator + " "):
                print(f"[Brain] detected follow-up question with pattern: '{indicator}'")
                return True
                
        # Check for questions that make little sense without context
        # These are often short questions with pronouns
        if len(question.split()) < 5:
            pronoun_indicators = ["it", "they", "them", "those", "these", "that", "this"]
            for pronoun in pronoun_indicators:
                if f" {pronoun} " in f" {question} ":
                    print(f"[Brain] detected follow-up question (short with pronoun): '{question}'")
                    return True
        
        return False
        
    def _is_opinion_question(self, question: str) -> bool:
        """Determine if the question is asking for opinions or preferences."""
        question = question.lower()
        
        # Detect common opinion/preference question patterns
        opinion_indicators = [
            # Direct questions about preferences
            
            "what do you like about", "how are you feeling", "what do you like", 
            "what do you enjoy", "what do you prefer", 
            "what kind of", "which do you prefer", "which do you like",
            "do you like", "do you enjoy", "do you prefer",
            "what's your opinion", "what is your opinion", "what are your thoughts on",

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

        
        # Check if any opinion indicators are present
        for indicator in opinion_indicators:
            question = question.replace("?", "").replace("'", "")
            if (indicator + " ") in (question + " "):
                print(f"[Brain] detected opinion question with pattern: '{indicator}'")
                return True
                
        return False
        
    def _generate_opinion(self, question: str) -> str:
        """Generate a personalized response to opinion/preference questions.
        
        This function now acts as an interactive chatbot, generating follow-up questions and
        maintaining a natural conversation flow for opinion-based discussions.
        """
        print(f"[Brain] generating opinion response for: '{question}'")
        
        # For shared brain, treat all opinion questions as first-time
        conversation_context = []
        is_first_opinion_question = True
        
        # Skip conversation context for shared brain
        if False:  # Disabled for shared brain
            # Count recent turns to determine if we're in a conversation
            opinion_count = 0
            for i in range(len(conversation_context)-1, -1, -1):
                # Skip the current question which is already marked as opinion
                if i == len(conversation_context)-1 and conversation_context[i]["role"] == "user":
                    continue
                # If we've gone back 3 turns (or hit beginning), stop counting
                if opinion_count >= 3:
                    break
                # Check if assistant response includes chatbot-like elements (questions, conversation markers)
                if conversation_context[i]["role"] == "assistant":
                    if "?" in conversation_context[i]["content"] or any(marker in conversation_context[i]["content"].lower() for marker in ["how about you", "what do you", "tell me more", "what about"]):
                        opinion_count += 1
            
            is_first_opinion_question = opinion_count == 0
            print(f"[Brain] detected {'new' if is_first_opinion_question else 'ongoing'} opinion conversation with {opinion_count} previous turns")

        # Add conversation context if this is a follow-up
        if not is_first_opinion_question and conversation_context:
            context_str = "\n\nCONVERSATION HISTORY:\n"
            for msg in conversation_context:
                role = "User" if msg["role"] == "user" else "You"
                context_str += f"{role}: {msg['content']}\n"
            prompt = context_str
        
        # Add specific instructions based on whether this is a new conversation or a follow-up
        if is_first_opinion_question:
            prompt = textwrap.dedent(
                f"""    
                The user has asked: "{question}"

                Your response:
                """
            )
        else:
            # For follow-up exchanges, focus more on natural conversation flow
            prompt += textwrap.dedent(
                f"""
                The user's latest message is: "{question}"
                
                Your response:
                """
            )

        try:
            # Generate the response with appropriate temperature for personality
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
            
            # Debug output
            print(f"[Brain] opinion prompt length: {len(prompt)} chars")
            print(f"[Brain] opinion response length: {len(response)} chars")
            
            # Extract just the response portion - remove any added quotation marks
            if "Your response:" in response:
                response = response.split("Your response:")[-1].strip()
                
            # Remove any surrounding quotes if present
            response = response.strip('"')
            
            print(f"[Brain] generated opinion response: {len(response)} chars")
            
            return response
            
        except Exception as e:
            print(f"[Brain] error generating opinion: {e}")
            
            # Fallback response if something goes wrong
            if is_first_opinion_question:
                return "Hey, that's a cool question! I'm still figuring out my thoughts on that. What kind of things do you like?"
            else:
                return "That's interesting! I'm still processing that one. What else is on your mind?"
        
    
    
    def _generate_answer(self, question: str, context_chunks: List[Dict], conversation_context: List[Dict] = None) -> str:
        
        """LLM final step: fuse context → natural-language answer."""

        
        if not context_chunks:
            
            print("[Brain] no context chunks available for answer generation")
            
            return "I don't have enough information to answer your question. I couldn't find relevant content in my memory or from web searches."

        
        print(f"[Brain] generating answer using {len(context_chunks)} context chunks")
        
        # Create short summaries of context chunks for logging
        def format_value_for_log(value):
            if isinstance(value, str) and len(value) > 50:
                return value[:50] + "..."
            elif isinstance(value, list):
                return f"[{len(value)} items]"
            else:
                return value
                
        log_context = [
            {k: format_value_for_log(v) for k, v in chunk.items()}
            for chunk in context_chunks
        ]
        print(f"[Brain] context summary: {json.dumps(log_context)}")
        
        
        # Format context for LLM prompt
        formatted_context = []
        for i, chunk in enumerate(context_chunks):
            # Start with the context number
            chunk_text = [f"CONTEXT {i+1}:"]
            
            # Add description if available
            if 'description' in chunk and chunk['description']:
                chunk_text.append(f"Description: {chunk['description']}")
            
            # Add bullet points if available
            if 'bullet_points' in chunk and isinstance(chunk.get('bullet_points'), list):
                points = []
                for point in chunk['bullet_points']:
                    if isinstance(point, str) and len(point) > 5:  # Skip very short points
                        points.append(f"  • {point}")
                if points:
                    chunk_text.append("Key points:")
                    chunk_text.extend(points)
            
            # Add other relevant fields
            for k, v in chunk.items():
                if k not in ('description', 'bullet_points') and isinstance(v, (str, int, float)):
                    chunk_text.append(f"{k.capitalize()}: {v}")
            
            # Join all parts with newlines
            formatted_context.append("\n".join(chunk_text))
        
        # Join all contexts with double newlines
        context_str = "\n\n".join(formatted_context)
        
        # Add conversation context if available
        conversation_str = ""
        if conversation_context and len(conversation_context) > 0:
            print(f"[Brain] including {len(conversation_context)} messages from conversation history")
            conversation_lines = []
            for msg in conversation_context:
                # Format chat history as a conversation
                speaker = "User" if msg["role"] == "user" else "Assistant"
                conversation_lines.append(f"{speaker}: {msg['content']}")
            conversation_str = "\n".join(conversation_lines)
            print(f"[Brain] conversation context length: {len(conversation_str)} chars")
        
        prompt = ""
        # Add conversation history context if available
        if conversation_str:
            prompt += textwrap.dedent(
                f"""
                CONVERSATION HISTORY:
                {conversation_str}
                """
            )
            
        # Continue with knowledge context and instructions
        prompt += textwrap.dedent(
            f"""
            KNOWLEDGE CONTEXT:
            {context_str}
            ANSWER:
            """ )

        # For follow-up questions, add specific instructions
        if conversation_context and len(conversation_context) > 0:
            prompt += textwrap.dedent(
                """
                This appears to be a follow-up question. Use the conversation history to understand the context of the question.
                Make sure your answer maintains continuity with the previous conversation.
                """
            )
            
        # Complete the prompt
        prompt += textwrap.dedent(
            f"""

            USER QUESTION: {question}
            
            ANSWER:"""
        )
        
        print("[Brain] sending prompt to LLM for final answer generation")
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

            print(f"[Brain] raw answer from LLM: {len(ans.strip())} chars")
            
            # Debug output for troubleshooting
            print(f"[Brain] final prompt length: {len(prompt)} chars")
            print(f"[Brain] raw answer length: {len(ans)} chars")
            
            # Extract only the answer portion
            answer_marker = "ANSWER:"
            if answer_marker in ans:
                # Use the text after the last occurrence of ANSWER:
                final_answer = ans.split(answer_marker)[-1].strip()
            else:
                # Try other possible markers
                markers = ["USER QUESTION:", "Answer:", "Response:"]
                for marker in markers:
                    if marker in ans:
                        parts = ans.split(marker)
                        if len(parts) > 1:
                            # Take the last part after this marker
                            final_answer = parts[-1].strip()
                            break
                else:
                    # If no markers found, use the whole response
                    final_answer = ans.strip()
            
            print(f"[Brain] extracted final answer: {len(final_answer)} chars")
            return final_answer
        except Exception as e:
            print(f"[Brain] error during answer generation: {e}")
            return "I encountered an error while trying to generate an answer based on the information I found. Please try asking your question again."


    # =====================================================================
    #                       💾  MEMORY PERSISTENCE
    # =====================================================================
    
    def _load_memory(self) -> bool:
        """Load the memory tree from a JSON file."""
        if not self.memory_file or not os.path.exists(self.memory_file):
            # No memory file to load
            return False
            
        try:
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                memory_dict = json.load(f)
                
            # Remove metadata before loading
            if "_meta" in memory_dict:
                save_time = memory_dict["_meta"].get("saved_at", "unknown")
                version = memory_dict["_meta"].get("version", "unknown")
                print(f"[Brain] loading memory saved at {save_time} (version {version})")
                del memory_dict["_meta"]
                
            # Create the memory tree from the dictionary
            self.memory = SupabaseTopicNode.from_dict(memory_dict)
            print(f"[Brain] successfully loaded memory from {self.memory_file}")
            return True
            
        except Exception as e:
            print(f"[Brain] ERROR loading memory from {self.memory_file}: {e}")
            # Start with a fresh memory tree
            self.memory = SupabaseTopicNode("Knowledge Base")
            return False

# -----------------------------------------------------------------------------
# Example usage (manual run)
# -----------------------------------------------------------------------------
    def explore_topics_autonomously(self,
                                      base_topics: Optional[List[str]] = None,
                                      *,
                                      max_depth: int = 3,
                                      breadth: int = 5,
                                      max_total_topics: int = 50,
                                      subtopics_per_topic: int = 5,
                                      delay: float = 0.5,
                                      temperature: float = 0.7,
                                      max_search_results: int = 5) -> List[str]:
        """💡 **Enhanced autonomous exploration**

        This upgraded routine aggressively expands the knowledge tree by performing a
        *breadth-first* crawl over a topic hierarchy that it generates on-the-fly with
        the LLM.  It repeatedly asks the model for *sub-topics* of each queued topic
        and dives up to *max_depth* levels deep, respecting *max_total_topics* so we
        don't explode our token quota.

        Parameters
        ----------
        base_topics : list[str] | None
            Seed topics to start from.  If *None*, we seed with a handful of very
            broad academic domains.
        max_depth : int, default 3
            How many hierarchical levels to descend (root = 0).
        breadth : int, default 5
            Maximum number of topics to fetch *per level*.
        max_total_topics : int, default 50
            Hard ceiling on how many individual topics we explore in a single run.
        subtopics_per_topic : int, default 5
            How many sub-topics to request from the LLM for each topic.
        delay : float, default 0.5
            Pause (seconds) between web calls so we don't hammer rate-limits.
        temperature : float, default 0.7
            Sampling temperature for LLM when brainstorming sub-topics.
        max_search_results : int, default 5
            Maximum number of websites to scrape for each topic.

        Returns
        -------
        list[str]
            A flat list of *every* topic question that was pushed through
            ``self.answer()`` during the session - useful for logs/stats.
        """

        print(f"[Brain] 🚀 starting enhanced autonomous exploration - target {max_total_topics} topics ...")

        explored: List[str] = []               # flat list of explored questions
        queue: List[Tuple[str, int]] = []      # (topic, depth)

        # ------------------------------------------------------------------
        # 1.  Seed queue
        # ------------------------------------------------------------------
        if base_topics:
            queue.extend([(t, 0) for t in base_topics[:breadth]])
            print(f"[Brain] seeding with user-supplied base topics: {base_topics[:breadth]}")
        else:
            default_roots = [
                "Science", "Technology", "Mathematics", "History", "Art",
                "Economics", "Psychology", "Biology", "Physics", "Computer Science"
            ]
            queue.extend([(t, 0) for t in random.sample(default_roots, k=min(breadth, len(default_roots)))])
            print(f"[Brain] seeding with default academic domains: {[q[0] for q in queue]}")

        # ------------------------------------------------------------------
        # 2.  Helper - expand a topic into subtopics using the LLM
        # ------------------------------------------------------------------
        def _generate_subtopics(topic: str, k: int) -> List[str]:
            prompt = textwrap.dedent(f"""
                List {k} highly specific sub-topics that a university-level
                researcher would study **within** the domain of "{topic}".
                • respond with one sub-topic per line, phrased as a direct
                  question beginning with *How*, *Why* or *What* so it can be
                  fed straight into a search engine.
            """)
            try:
                raw = self.gen(prompt, max_new_tokens=64, temperature=temperature)[0]["generated_text"]
                lines = [re.sub(r"^[•\-\d\.\s]*", "", l).strip() for l in raw.splitlines()]
                # keep unique, non-empty, question-like lines
                sub = []
                for l in lines:
                    if l and l[-1] == '?' and l not in sub:
                        sub.append(l)
                    if len(sub) >= k:
                        break
                return sub
            except Exception as e:
                print(f"[Brain] sub-topic generation failed for '{topic}': {e}")
                return []

        # ------------------------------------------------------------------
        # 3.  BFS over topic hierarchy
        # ------------------------------------------------------------------
        while queue and len(explored) < max_total_topics:
            topic, d = queue.pop(0)
            question = f"Tell me about {topic}" if not topic.endswith('?') else topic

            print(f"[Brain] ▸ exploring (depth {d}) - {question}")
            try:
                self.answer(question, max_search_results=max_search_results) 
                explored.append(question)
                self._save_memory()
            except Exception as e:
                print(f"[Brain] ⚠️ exploration error: {e}")

            # stop descending if depth limit reached or global budget exhausted
            if d >= max_depth or len(explored) >= max_total_topics:
                continue

            # brainstorm sub-topics and enqueue them
            children = _generate_subtopics(topic, subtopics_per_topic)
            random.shuffle(children)            # add variety
            for child in children[:breadth]:
                if len(explored) + len(queue) >= max_total_topics:
                    break
                queue.append((child, d + 1))

            time.sleep(delay)

        print(f"[Brain] 🏁 exploration complete - {len(explored)} topics added to memory")
        return explored

    async def process_search_results_parallel(self, search_hits, user_question: str):
        """Process search results in parallel - scraping and summarizing concurrently"""
        
        async def process_single_hit(hit: Dict, index: int) -> Optional[Dict]:
            """Process a single search hit: scrape content"""
            print(f"[Brain] processing result {index+1}/{len(search_hits)}: {hit['title']}")
            print(f"[Brain] URL: {hit['url']}")
            
            raw = await asyncio.get_event_loop().run_in_executor(None, self._scrape, hit["url"])
            
            if not raw:
                print(f"[Brain] no content scraped from {hit['url']}")
                return None
                
            print(f"[Brain] scraped {len(raw)} chars from {hit['url']}")
            
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
                print(f"[Brain] error processing result {i+1}: {str(result)}")
            elif result is not None:
                successful_results.append(result)
        
        print(f"[Brain] successfully processed {len(successful_results)} out of {len(search_hits)} search results")
        
        # If no successful results, return early
        if not successful_results:
            return "My bad bro I couldn't find any information on that topic. You can try rephrasing your question."
        
        # Prepare the prompt with multiple websites
        formatted_websites = ""
        for i, item in enumerate(successful_results, 1):
            formatted_websites += f"website {i}: {item['content']}\n\n"
        
        # Prepare summarization prompt
        prompt = f"{formatted_websites}"
        
        print("[Brain] sending multi-website summarization prompt")
        print(f"[Brain] processing multi-website summary")

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
            print(f"[Brain] multi-website summary response {response}")
            # Parse the JSON response
            structured_results = json.loads(response)
            
            # Store each summary
            async def store_single_result(summary: Dict) -> bool:
                """Store a single processed result"""
                try:
                    # Generate topic path
                    path = await asyncio.get_event_loop().run_in_executor(
                        None, self._generate_topic_path, user_question, summary.get("main_idea", "Unknown"), summary
                    )
                    
                    print(f"[Brain] storing under path: {'/'.join(path)}")
                    
                    await asyncio.get_event_loop().run_in_executor(
                        None, self._insert_memory, path, summary, summary.get("source", "")
                    )
                    
                    return True
                except Exception as e:
                    print(f"[Brain] failed to store result: {str(e)}")
                    return False
            
            print(f"[Brain] storing {len(structured_results)} summary results")

            if structured_results:
                store_tasks = [store_single_result(item) for item in structured_results]
                store_results = await asyncio.gather(*store_tasks, return_exceptions=True)
                
                successful_stores = sum(1 for result in store_results if result is True)
                print(f"[Brain] stored information from {successful_stores}/{len(structured_results)} search results")

            # Retrieve again with fresh data
            print("[Brain] retrieving context with fresh data...")
            context = self._retrieve_context(user_question, k=5)
            print(f"[Brain] found {len(context)} contexts for final answer generation")
            
            print("[Brain] generating final answer...")
            response = self._generate_answer(user_question, context, [])
            print(f"[Brain] answer generated ({len(response)} chars)")
            
            return response
            
        except Exception as e:
            print("[Brain] failed to search web: ", e)
            return "My bad bro I couldn't find any information on that topic. You can try rephrasing your question."

# -----------------------------------------------------------------------------
# Example usage (manual run)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
     brain = Brain()
     
     # Test opinion generation (interactive chatbot)
     print("\n>>> Starting opinion conversation...")
     opinion = brain.answer("What's your favorite type of music?")
     print("\n>>>", opinion)
     
     # Continue the conversation
     follow_up = brain.answer("That's cool! I like rock music. What do you think about movies?")
     print("\n>>>", follow_up)
     
     # Test factual question with multiple websites (aggregated search)
     print("\n>>> Asking a factual question with 3 search results...")
     factual = brain.answer("How are black holes formed?", max_search_results=3)
     print("\n>>>", factual)
     
