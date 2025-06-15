from typing import Any, Dict, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import timedelta
import os
from dotenv import load_dotenv
from flask_jwt_extended import JWTManager
from LLM import Brain
import jwt

load_dotenv()

app = Flask(__name__)
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"], "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"]}})

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config["JWT_ALGORITHM"] = "HS256"
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///users.db')
app.config['JWT_SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)
app.config['MISTRAL_API_KEY'] = os.getenv('MISTRAL_API_KEY', None)

jwt_manager = JWTManager(app)

def get_authenticated_user() -> str:
    """
    Returns the uuid (sub) if valid

    Returns:
        str: The user uuid (sub) if valid, None otherwise.
    """
    auth_header = request.headers.get('Authorization', None)
    if auth_header and auth_header.startswith("Bearer "):
        try:
            token = auth_header.split(" ")[1]
            user = verify_supabase_jwt(token)
            if user:
                return user.get('sub', None)
        except Exception:
            print("No valid Authorization header found")
    return None

def verify_supabase_jwt(token) -> Optional[Dict[str, Any]]:
    """
    Verify a JWT token from Supabase.

    Args:
        token (str): The JWT token to verify.

    Returns:
        Optional[Dict[str, Any]]: The decoded token if valid, None otherwise.
    """
    try:
        jwt_secret = os.environ.get("SUPABASE_JWT_SECRET")
        if not jwt_secret:
            print("Error: SUPABASE_JWT_SECRET not configured")
            return None
            
        decoded = jwt.decode(
            token,
            jwt_secret,
            algorithms=["HS256"],
            options={"verify_aud": False} #Don't verify audience in dev
        )
        return decoded
    except jwt.ExpiredSignatureError:
        print("JWT token expired")
        return None
    except jwt.InvalidTokenError as e:
        print(f"Invalid JWT token: {e}")
        return None
    except Exception as e:
        print(f"JWT verification error: {e}")
        return None


global_brain = None

def get_global_brain() -> Brain:
    """
    shared brain instance for all users to make hozie get smarter over time from all user queries instead of a per user brain that will only learn from that one user

    Returns:
        Brain: The global brain instance.
    """
    global global_brain
    if global_brain is None:
        try:
            global_brain = Brain()
        except Exception as e:
            print(f"Failed to create global brain: {e}")
            raise e
    
    return global_brain

@app.after_request
def add_cors_headers(response):
    """
    Add CORS headers to the response.

    Args:
        response: The Flask response object.

    Returns:
        response: The modified response object with CORS headers.
    """
    origin = request.headers.get('Origin', '')
    response.headers.add('Access-Control-Allow-Origin', origin)
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-Requested-With')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response


@app.route('/api/chat', methods=['POST'])
def chat() -> Any:
    """
    Only allow users to chat if they are authenticated (They have to verify email too) then return a response from the brain instance.

    Returns:
        Any: JSON response containing the reply from the brain instance or an error message.
    """
    user_id = get_authenticated_user()
    if not user_id:
        print("Chat request rejected: Authentication failed")
        return jsonify({
            'reply': 'I got an authentication error. Make sure your email is verified. If it is then try logging in again bro.',
            'error': 'Authentication required'
        }), 200
    
    print(f"User authenticated: {user_id}")

    try:
        data = request.get_json()
        if not data or 'message' not in data:
            print("Chat request rejected: No message provided in request")
            return jsonify({'error': 'No message provided'}), 400
        question = data['message']
            
        print(f"Received chat message from {user_id}: {question[:30]}...")
        
    except Exception as e:
        print(f"Request parsing error: {e}")
        return jsonify({'error': 'Invalid request format'}), 400

    try:
        brain_instance = get_global_brain()
    except Exception as e:
        print(f"Failed to get global brain: {e}")
        return jsonify({'reply': "Hozie is not here right now. He's probably out surfing. Check back in a few."}), 503

    try:
        print(f"Processing message for user {user_id} with shared brain...")
        reply = brain_instance.answer(question)
        return jsonify({
            'reply': reply,
            'user': user_id
        })
    except Exception as e:
        import traceback
        print(f"Chat error for user {user_id}: {e}")
        print(traceback.format_exc())
        
        return jsonify({
            'reply': 'What? Sorry dude I spaced out on that one. Can you ask me again?',
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health() -> Any:
    """
    Health check endpoint to verify if the backend is running and the brain instance is active.

    Returns:
        Any: JSON response indicating the health status of the backend and the brain instance.
    """
    return jsonify({
        'status': 'healthy',
        'brain_instance_active': global_brain is not None,
    })


@app.route('/api/queue_status', methods=['GET'])
def queue_status() -> Any:
    """
    queue if theres a lot of users chatting at once (not really implemented fully since I don't have that many users yet lol)
    Returns the queue status for the authenticated user.

    Returns:
        Any: JSON response containing the user's queue position and status.
    """
    user_id = get_authenticated_user()
    if not user_id:
        return jsonify({'error': 'Authentication required'}), 401
    
    return jsonify({
        'user': user_id,
        'queue_position': 0,
        'is_active': True,
        'stats': {'brain_instance': 'active'}
    })

if __name__ == '__main__':
    print("Starting backend...")
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)
