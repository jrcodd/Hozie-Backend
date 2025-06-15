
## Hozie Backend Service - https://hozie.netlify.app
<img width="590" alt="Screenshot 2025-05-30 at 2 54 24 PM" src="https://github.com/user-attachments/assets/7dba05b1-081b-4dec-81be-435767230217" />

<img width="1419" alt="Screenshot 2025-05-30 at 2 56 57 PM" src="https://github.com/user-attachments/assets/2c32bcf9-6ec1-4b4a-a0bf-fec4e394a7e9" />
<img width="1427" alt="Screenshot 2025-05-28 at 10 38 00 AM" src="https://github.com/user-attachments/assets/fc7772b9-3654-486c-9bc6-d4d144536441" />
This repository contains the backend service for Hozie, a full-stack AI chatbot application. The service is built with Python and Flask and is responsible for user authentication, API routing, and orchestrating the core AI logic for conversational responses.

The core of this service is the `Brain` class, which implements a sophisticated cognitive architecture. This allows Hozie to maintain a persistent, shared "global brain" that learns and grows from all user interactions, storing its knowledge in a structured tree within a Supabase database.

## Core Features & Architecture

This backend is designed to support a scalable and intelligent conversational AI through a series of specialized modules.

*   **RESTful API**: A Flask-based API serves as the interface between the client application and the core AI logic[1].
*   **JWT Authentication**: Secures the API using JWTs issued by Supabase. The backend validates these tokens to ensure only authenticated users can interact with the chatbot[1].
*   **Persistent Global Memory**: Unlike per-session chatbots, Hozie uses a singleton `Brain` instance that connects to a shared Supabase backend. This allows its knowledge base to grow and improve over time across all users[1][2].
*   **RAG Pipeline**: The system uses a dynamic Retrieval-Augmented Generation pipeline:
    1.  **Memory-First Search**: Attempts to answer questions using a custom heuristic search on its existing knowledge tree[3].
    2.  **Dynamic Web Search**: If memory is insufficient, it uses the Brave Search API to find relevant, up-to-date information on the web[3].
    3.  **Asynchronous Scraping**: Leverages `asyncio` to scrape content from multiple web sources in parallel for maximum efficiency[3].
    4.  **LLM-Powered Summarization**: Uses a Mistral agent to process scraped content into a structured JSON format (`main_idea`, `description`, `bullet_points`)[3].
    5.  **Hierarchical Knowledge Storage**: Generates a logical topic path (e.g., `["Science", "Astronomy", "Black Holes"]`) and stores the summarized knowledge in a tree structure within Supabase using `SupabaseTopicNode`[3][2].
*   **Advanced Conversational Handling**: Includes dedicated logic to identify and respond appropriately to opinion-based questions and conversational follow-ups[3].

## Tech Stack

*   **Backend Framework**: Flask[1]
*   **AI & LLMs**: `mistralai` SDK (leveraging Mistral Agents for specific tasks)[3]
*   **Database**: Supabase (PostgreSQL)
    *   `memory_nodes` table for the knowledge tree (`SupabaseTopicNode`)[2].
    *   `chat_history` table for persisting conversations (`SupabaseChatHistory`)[4].
*   **Authentication**: Flask-JWT-Extended, PyJWT (for Supabase token verification)[1]
*   **Web Interaction**: Brave Search API, Requests, BeautifulSoup4[3]
*   **Asynchronous Operations**: `asyncio` for parallel web processing[3]
*   **Environment Management**: `python-dotenv`[1]

## API Endpoints

| Method | Endpoint          | Description                                                    | Auth Required |
| :----- | :---------------- | :------------------------------------------------------------- | :------------ |
| `POST`   | `/api/chat`       | Main endpoint for submitting a user message and getting a reply. | Yes           |
| `GET`    | `/health`         | Health check to verify that the server and Brain are active.   | No            |
| `GET`    | `/api/queue_status` | Placeholder endpoint for future queue management.              | Yes           |

## Getting Started

Follow these instructions to set up and run the backend service on your local machine.

### 1. Prerequisites

*   Python 3.9+ (as noted in `requirements.txt`)[5]
*   Git
*   A Supabase project.
*   API Keys for:
    *   Mistral AI
    *   Brave Search API

### 2. Installation & Setup

First, clone the repository to your local machine:
```bash
git clone https://github.com/YourUsername/hozie-backend.git
cd hozie-backend
```

Next, create a virtual environment and install the required dependencies from `requirements.txt`[5]:
```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Supabase Table Setup

Before running the application, you need to create two tables in your Supabase project via the SQL Editor:

**`memory_nodes` Table (for the knowledge tree):**
```sql
CREATE TABLE public.memory_nodes (
  id UUID PRIMARY KEY,
  name TEXT NOT NULL,
  parent_id UUID REFERENCES public.memory_nodes(id) ON DELETE CASCADE,
  description TEXT,
  memories JSONB,
  metadata JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

**`chat_history` Table (for conversation logs):**
```sql
CREATE TABLE public.chat_history (
  id UUID PRIMARY KEY,
  user_id UUID NOT NULL,
  message_type TEXT NOT NULL,
  content TEXT,
  "timestamp" TIMESTAMPTZ DEFAULT NOW() NOT NULL,
  metadata JSONB
);
```

### 4. Environment Configuration

This project requires API keys and database credentials. Create a `.env` file in the root directory by copying the example file:
```bash
cp .env.example .env
```

Now, edit the `.env` file with your credentials. These are all required by the code[1][6][3].
```ini
# .env - Hozie Backend Configuration

# Flask Settings
SECRET_KEY="your-strong-flask-secret-key"

# API Keys
MISTRAL_API_KEY="sk-..."
BRAVE_SEARCH_API_KEY="your-brave-api-key"

# Supabase Credentials
SUPABASE_URL="https://.supabase.co"
SUPABASE_SERVICE_ROLE_KEY="your-supabase-service-role-key" # For server-side operations
SUPABASE_JWT_SECRET="your-supabase-jwt-secret-from-auth-settings"
```

## Running the Service

With the virtual environment activated and the `.env` file configured, you can run the cli.py file to interact with Hozie or you can run the Flask development server:
```bash
python3 cli.py
```
or 
```bash
flask run
```
or 
```bash
python3 run.py
```

The API will now be available at `http://127.0.0.1:5000`.

To make the server accessible on your local network (e.g., for testing with a mobile client), use:
```bash
flask run --host=0.0.0.0
```

## Future Work & Improvements

Based on notes in the codebase, potential areas for future development include:
*   **Enhanced Search Queries**: Implement logic to generate a list of search queries (e.g., with `doctype:pdf`) to improve coverage for research-based questions[3].
*   **Refined Relevance Threshold**: Test and tune the `0.07` relevance threshold in the `_is_context_relevant` function to optimize the trade-off between memory recall and web search frequency[3].
*   **Full Follow-up Context**: Fully implement the logic to pass previous conversation history to the final answer generation prompt for follow-up questions[3].

## License

This project is licensed under the MIT License.

[1] app.py
[2] supabase_topic_node.py
[3] LLM.py
[4] supabase_chat_history.py
[5] requirements.txt
[6] supabase_client.py
[7] chat_history.py
[8] supabase_user_client.py
