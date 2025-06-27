import os
import openai
import pickle
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import tiktoken
import json
import re
import logging
import datetime

# -*- coding: utf-8 -*-

# Load environment variables
load_dotenv()

# Load model comparisons data
with open("product_files/comparisons.json", "r") as f:
    model_comparisons = json.load(f)

# Load dealers data
with open("product_files/dealers.json", "r") as f:
    dealers_data = json.load(f)

# Load general answers data
with open("product_files/general_answers.json", "r") as f:
    general_answers = json.load(f)

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Initialize tokenizer
GPT_MODEL = "gpt-4.1"
EMBEDDING_MODEL = "text-embedding-ada-002"
#encoding = tiktoken.encoding_for_model(GPT_MODEL)
encoding = tiktoken.get_encoding("cl100k_base")

# Global token usage tracking
token_usage = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "embedding_tokens": 0
}

def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    return len(encoding.encode(text))

from collections import defaultdict

chat_history = defaultdict(list)  # {ip_address: [summary1, summary2, ...]}

# === CONFIG ===
openai.api_key = ""
VECTOR_META_PATH = "metadata.pkl"
VECTOR_EMBEDDINGS_PATH = "vector_embeddings.pkl"
EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI model for embeddings
MAX_TOKENS = 500
# === FASTAPI SETUP ===
app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# === MODELS ===
class ChatRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = "default"
    max_results: Optional[int] = 3
    include_sources: Optional[bool] = True

class SearchResult:
    def __init__(self, content, source=None):
        self.content = content
        self.source = source

def get_embedding(text: str, engine: str = EMBEDDING_MODEL) -> list:
    global token_usage
    text = text.replace("\n", " ")  # Clean input
    # Count embedding tokens
    num_tokens = count_tokens(text)
    token_usage["embedding_tokens"] += num_tokens
    logger.info(f"Embedding Request - Text: {text[:100]}... ({num_tokens} tokens)")
    
    try:
        response = openai.Embedding.create(
            input=[text],
            model=engine
        )
        return response["data"][0]["embedding"], num_tokens
    except Exception as e:
        logger.error(f"Error getting embedding: {str(e)}")
        raise e
        
def load_model_chunks(data: dict) -> dict:
    model_chunks = {}
    for model_name, model_info in data.items():
        flat = [f"Model: {model_name}"]
        # details, description, values are all dicts
        for section in ("details", "description", "values"):
            sec = model_info.get(section, {})
            if isinstance(sec, dict):
                flat.append(f"\n[{section.upper()}]")
                for k, v in sec.items():
                    flat.append(f"{k}: {v}")
        model_chunks[model_name] = "\n".join(flat)
    return model_chunks

# === UTILITIES ===
def chunk_text(text, max_tokens=100, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap
    return chunks


def read_product_files(directory="product_files"):
    files = []
    if not os.path.exists(directory):
        return []
    for filename in os.listdir(directory):
        if filename.endswith(('.txt', '.md', '.json')):
            try:
                with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
                    content = f.read()
                    files.append(SearchResult(content=content, source=filename))
            except Exception:
                continue
    return files
def build_vector_index():
    documents = []
    metadata = []
    chunk_embeddings = []

    # First, load the JSON files directly
    json_files = {
        "comparisons.json": model_comparisons,
        #"dealers.json": dealers_data,
        "general_answers.json": general_answers
    }
    # Process JSON files
        # Process JSON files
       # Process JSON files
   # Process JSON files
    for filename, data in json_files.items():
        try:
            logger.info(f"[INDEX BUILD] processing {filename}, type(data)={type(data)}")
            if filename == "comparisons.json":
                # if somehow data is a raw JSON string, parse it
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                        logger.info("[INDEX BUILD] parsed comparisons.json string into dict")
                    except Exception as e:
                        logger.error(f"[INDEX BUILD] cannot parse comparisons.json: {e}")
                        continue

                if not isinstance(data, dict):
                    logger.error(f"[INDEX BUILD] unexpected comparisons.json type: {type(data)}; skipping")
                    continue

                # now flatten each model
                chunks_dict = load_model_chunks(data)
                logger.info(f"[INDEX BUILD] Loaded {len(chunks_dict)} model-chunks from {filename}")
                for model_name, chunk in chunks_dict.items():
                    embedding, tokens = get_embedding(chunk)
                    logger.info(f"[INDEX BUILD] {filename}?{model_name}: {tokens} tokens")
                    chunk_embeddings.append({
                        "embedding": embedding,
                        "chunk": chunk,
                        "source": filename,
                        "model": model_name
                  })
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            continue



    # Process text files
    for filename in os.listdir("product_files"):
        if filename.endswith((".txt", ".md")) and filename not in json_files:
            path = os.path.join("product_files", filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if not content:
                        continue

                    # Split into chunks
                    chunks = chunk_text(content, max_tokens=500, overlap=100)
                    for chunk in chunks:
                        embedding, tokens = get_embedding(chunk, engine=EMBEDDING_MODEL)
                        logger.info(f"Chunk Embedding - Source: {filename}, Tokens: {tokens}")
                        chunk_embeddings.append({
                            "embedding": embedding,
                            "chunk": chunk,
                            "source": filename
                        })
            except Exception as e:
                logger.error(f"Error processing file {filename}: {str(e)}")
                continue

    # Save embeddings
    with open("chunk_embeddings.pkl", "wb") as f:
        pickle.dump(chunk_embeddings, f)

    logger.info(f"Vector index built successfully with {len(chunk_embeddings)} total chunks")


def get_relevant_chunks(query, k=10, max_tokens=MAX_TOKENS):
    # sourcery skip: inline-immediately-returned-variable
    query_embedding, embedding_tokens = get_embedding(query, engine=EMBEDDING_MODEL)
    token_usage["embedding_tokens"] += embedding_tokens

    with open("chunk_embeddings.pkl", "rb") as f:
        all_chunks = pickle.load(f)

    similarities = []
    for chunk_info in all_chunks:
        similarity = cosine_similarity([query_embedding], [chunk_info["embedding"]])[0][0]
        similarities.append((similarity, chunk_info))

    similarities.sort(key=lambda x: x[0], reverse=True)

    selected_chunks = []
    total_tokens = 0
    for similarity, chunk_info in similarities:
        chunk_tokens = len(chunk_info["chunk"].split())
        if total_tokens + chunk_tokens <= max_tokens:
            selected_chunks.append(chunk_info)
            total_tokens += chunk_tokens
        else:
            break

    results = []
    for chunk in selected_chunks:
        sr = SearchResult(content=chunk["chunk"], source=chunk["source"])
        # assign the model name (or None)
        sr.model = chunk.get("model")
        results.append(sr)
    return results


def summarize_message(text: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # summarization model
            messages=[
                {"role": "system", "content": "Summarize the following message in one sentence."},
                {"role": "user", "content": text}
            ],
            temperature=0.3,
            max_tokens=40
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"Summarization fallback: {str(e)}")
        return text[:60].strip().replace("\n", " ") + "..." if len(text) > 60 else text
        
# === ROUTES ===
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

from collections import defaultdict

# In-memory chat history
chat_history = defaultdict(list)

def summarize_message(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": "Summarize the following user message in one sentence:"},
                {"role": "user", "content": text}
            ],
            temperature=0.3,
            max_tokens=30
        )
        summary = response["choices"][0]["message"]["content"]
        return summary.strip()
    except Exception as e:
        # Fallback to basic truncation
        return text[:60].strip().replace("\n", " ") + "..." if len(text) > 60 else text

def extract_specific_models(query: str) -> list:
    """Extract specific model names from a comparison query."""
    query_lower = query.lower()
    models = []
    
    # Common model prefixes with different possible formats
    prefixes = ["tx", "kr", "qc"]
    
    # First check for exact "TX7 and TX12" or "TX7 vs TX12" patterns
    combined_pattern = r'(?:compare\s+)?([a-zA-Z]{2,3})[-\s]?(\d+)(?:\s+(?:and|vs|or|against)\s+)([a-zA-Z]{2,3})[-\s]?(\d+)'
    combined_match = re.search(combined_pattern, query_lower)
    if combined_match:
        prefix1, number1, prefix2, number2 = combined_match.groups()
        model1 = f"{prefix1.upper()}{number1}"
        model2 = f"{prefix2.upper()}{number2}"
        models.extend([model1, model2])
        return models
    
    # Look for models mentioned with numbers (like TX7, KR30)
    for prefix in prefixes:
        # Pattern to match models like TX7, TX-7, TX 7, etc.
        pattern = fr'{prefix}[-\s]?(\d+)'
        matches = re.finditer(pattern, query_lower)
        for match in matches:
            number = match.group(1)
            # Standardize model name format (e.g., TX7, KR30)
            model_name = f"{prefix.upper()}{number}"
            models.append(model_name)
    
    return models

def handle_specific_model_comparison(query: str) -> bool:
    """Check if query is a direct model comparison request like 'Compare TX7 and TX12'"""
    comparison_terms = ["compare", "vs", "versus", "against", "and"]
    query_lower = query.lower()
    
    # Check if this is a comparison query
    if not any(term in query_lower for term in comparison_terms):
        return False
    
    # Extract models from the query
    specific_models = extract_specific_models(query)
    
    # Need at least 2 models for comparison
    return len(specific_models) >= 2
    
def extract_specific_models(query: str) -> list:
    """
    Pulls out any model names in the form """
    query_lower = query.lower()
    models = []

    # First, catch two-model comparisons like "KR30 vs KR44"
    combined_pattern = (
        r'(?:compare\s+)?'          # optional "compare"
        r'([a-zA-Z]{2,3})[-\s]?(\d+)'  # prefix+number
        r'(?:\s+(?:and|vs|or|against)\s+)' 
        r'([a-zA-Z]{2,3})[-\s]?(\d+)'  # second prefix+number
    )
    m = re.search(combined_pattern, query_lower)
    if m:
        p1,n1,p2,n2 = m.groups()
        return [f"{p1.upper()}{n1}", f"{p2.upper()}{n2}"]

    # Otherwise pick up any standalone mentions
    for prefix in ("tx","kr","qc"):
        for match in re.finditer(fr'{prefix}[-\s]?(\d+)', query_lower):
            num = match.group(1)
            models.append(f"{prefix.upper()}{num}")

    return models

def get_model_comparison_data(query: str) -> dict:
    """Extract relevant model comparison data based on the query."""
    relevant_data = {
        "models": {},
        "series_info": {}
    }
    
    # Convert query to lowercase for case-insensitive matching
    query_lower = query.lower()
    
    # Extract power requirement if present
    power_requirement = None
    power_match = re.search(r'(\d+)\s*(?:kw|kilowatt)', query_lower)
    if power_match:
        power_requirement = int(power_match.group(1))
    
    # Check for specific model comparison
    if "compare" in query_lower or "vs" in query_lower or "difference" in query_lower or "and" in query_lower:
        # Extract specific models first
        specific_models = extract_specific_models(query)
        
        if len(specific_models) >= 2:
            # We have specific models to compare
            for model_name in specific_models:
                prefix = re.match(r'([A-Za-z]+)', model_name).group(1)
                series_name = f"{prefix} Models"
                
                # Find this model in the appropriate series
                if series_name in model_comparisons:
                    for model_key, model_data in model_comparisons[series_name].items():
                        if model_key.replace(" ", "") == model_name or model_key == model_name:
                            relevant_data["models"][model_name] = model_data
                            break
    
    # If no specific models found or not enough for comparison, handle power-based matching
    if len(relevant_data["models"]) < 2:
        # Check for specific series mentions
        series_keywords = {
            "tx": "TX Models",
            "kr": "KR Models",
            "qc": "QC Models"
        }
        
        # If we have a power requirement, find the best matching model
        if power_requirement:
            best_match = None
            best_match_power = float('inf')
            
            for series in series_keywords.values():
                if series in model_comparisons:
                    for model_key, model_data in model_comparisons[series].items():
                        model_power = model_data["details"].get("KW", 0)
                        if model_power >= power_requirement and model_power < best_match_power:
                            best_match = (model_key, model_data)
                            best_match_power = model_power
            
            if best_match:
                model_key, model_data = best_match
                relevant_data["models"][model_key] = model_data
        else:
            # No power requirement, check for series mentions
            for keyword, series in series_keywords.items():
                if keyword in query_lower:
                    if series in model_comparisons:
                        # Create clean copies of all models in the series
                        for model_key, model_data in model_comparisons[series].items():
                            clean_data = {
                                "details": model_data["details"].copy(),
                                "description": model_data["description"].copy(),
                                "values": model_data["values"].copy()
                            }
                            relevant_data["models"][model_key] = clean_data
                    
            # If no specific series mentioned, include all for general comparison queries
            if len(relevant_data["models"]) == 0 and ("compar" in query_lower or "vs" in query_lower or "difference" in query_lower):
                for series in series_keywords.values():
                    if series in model_comparisons:
                        # Create clean copies of all models in the series
                        for model_key, model_data in model_comparisons[series].items():
                            clean_data = {
                                "details": model_data["details"].copy(),
                                "description": model_data["description"].copy(),
                                "values": model_data["values"].copy()
                            }
                            relevant_data["models"][model_key] = clean_data
    
    return relevant_data

def extract_postal_code(text: str) -> Optional[str]:
    """Extract postal/zip code from text using regex patterns."""
    # Canadian postal code pattern (A1A 1A1)
    ca_pattern = r'[A-Za-z]\d[A-Za-z]\s?\d[A-Za-z]\d'
    # US ZIP code pattern (12345 or 12345-6789)
    us_pattern = r'\b\d{5}(?:-\d{4})?\b'
    # Indian PIN code pattern (6 digits)
    in_pattern = r'\b\d{6}\b'
    
    # Try Canadian postal code first
    ca_match = re.search(ca_pattern, text)
    if ca_match:
        return ca_match.group().upper().replace(" ", "")
    
    # Try US ZIP code
    us_match = re.search(us_pattern, text)
    if us_match:
        return us_match.group()
    
    # Try Indian PIN code
    in_match = re.search(in_pattern, text)
    if in_match:
        return in_match.group()
    
    return None

def calculate_postal_similarity(input_code: str, dealer_code: str) -> float:
    """Calculate similarity between two postal codes based on prefix match and character overlap."""
    input_code = input_code.strip().replace(" ", "").upper()
    dealer_code = dealer_code.strip().replace(" ", "").upper()

    # Exact match gets perfect score
    if input_code == dealer_code:
        return 1.0

    # If both are numeric
    if input_code.isdigit() and dealer_code.isdigit():
        # Use inverse of absolute difference (normalized)
        diff = abs(int(input_code) - int(dealer_code))
        return max(0, 1 - (diff / 10000))  # Tunable divisor

    # For alphanumeric (Canadian) codes
    min_len = min(len(input_code), len(dealer_code))
    match_len = 0
    for i in range(min_len):
        if input_code[i] == dealer_code[i]:
            match_len += 1
        else:
            break

    return match_len / max(len(dealer_code), 1)

def find_nearest_dealers(postal_code: str, num_dealers: int = 2) -> list:
    """Find the nearest dealers based on postal code similarity."""
    if not postal_code:
        return []
    
    # Calculate similarity scores for all dealers
    dealer_scores = []
    for dealer in dealers_data["dealers"]:
        if "location" in dealer:  # Make sure dealer has a location
            similarity = calculate_postal_similarity(postal_code, dealer["location"])
            dealer_scores.append((similarity, dealer))
    
    # Sort by similarity score (descending) and return top N dealers
    dealer_scores.sort(reverse=True, key=lambda x: x[0])
    return [dealer for _, dealer in dealer_scores[:num_dealers]]

def is_dealer_query(text: str) -> bool:
    """Check if the query is about finding dealers."""
    dealer_keywords = [
        "dealer", "dealers", "location", "nearest", "closest", "near me",
        "where can i buy", "where to buy", "purchase location", "near to me"
    ]
    text_lower = text.lower()
    
    # If it's just a postal code, don't treat it as a new dealer query
    if extract_postal_code(text) and len(text.strip()) <= 7:
        return False
        
    return any(keyword in text_lower for keyword in dealer_keywords)

@app.post("/ask")
async def ask_bot(request: Request):
    global token_usage
    try:
        body = await request.json()
        chat_request = ChatRequest(**body)
        question = chat_request.prompt.strip()
        session_id = chat_request.session_id or "default"

        # Push the new user message onto their history
        chat_history[session_id].append({"role": "user", "content": question})

        # Handle empty questions
        if not question:
            return {
                "response": json.dumps({
                    "Description": "Please provide a question or message.",
                    "Links": [],
                    "Questions": ["What would you like to know about our products?"]
                })
            }

        # Check for simple greetings first
        greeting_words = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
        if question.lower().strip() in greeting_words:
            return {
                "response": json.dumps({
                    "Description": "Hi! How can I help you with Baumalight products today? I can provide information about our generators, help you compare models, or find a dealer near you.",
                    "Links": [
                        '<a href="https://baumalight.com/product/generators/tx-models" target="_blank">TX Models</a>',
                        '<a href="https://baumalight.com/product/generators/kr-models" target="_blank">KR Models</a>',
                        '<a href="https://baumalight.com/product/generators/qc-singlephase" target="_blank">QC Models</a>'
                    ],
                    "Questions": []
                })
            }

        # Check for dealer query or postal code
        is_dealer = is_dealer_query(question)
        postal_code = extract_postal_code(question)

        if is_dealer or postal_code:
            if not postal_code:
                return {
                    "response": json.dumps({
                        "Description": "I can help you find the nearest dealers. Could you please provide your postal/ZIP code?",
                        "Links": [
                            '<a href="https://baumalight.com/product/locator" target="_blank">Dealer Locator</a>'
                        ],
                        "Questions": []
                    })
                }

            nearest_dealers = find_nearest_dealers(postal_code)
            if not nearest_dealers:
                return {
                    "response": json.dumps({
                        "Description": f"I notice you're looking from {postal_code}. Currently, our dealers are located in North America. Please visit our dealer locator page for more information.",
                        "Links": [
                            '<a href="https://baumalight.com/product/locator" target="_blank">Dealer Locator</a>'
                        ],
                        "Questions": []
                    })
                }

            dealer_info = format_dealer_response(nearest_dealers, postal_code)
            return {"response": json.dumps(dealer_info)}

        # Check for model comparison
        is_comparison = handle_specific_model_comparison(question)
        if is_comparison:
            models_to_compare = extract_specific_models(question)
            comparison_response = generate_model_comparison(models_to_compare)
            if comparison_response:
                return {"response": json.dumps(comparison_response)}

        # Get model data and relevant chunks
        model_data = get_model_comparison_data(question)
        relevant_chunks = get_relevant_chunks(question, k=3)

        if not relevant_chunks and not model_data["models"]:
            return {
                "response": json.dumps({
                    "Description": "I can help you with information about our generator models, product comparisons, or finding a dealer. What would you like to know specifically?",
                    "Links": [
                        '<a href="https://baumalight.com/product/generators/tx-models" target="_blank">TX Models</a>',
                        '<a href="https://baumalight.com/product/generators/kr-models" target="_blank">KR Models</a>',
                        '<a href="https://baumalight.com/product/generators/qc-singlephase" target="_blank">QC Models</a>'
                    ],
                    "Questions": []
                })
            }

        # Build context and get response from OpenAI
        context = build_context(relevant_chunks)
        bot_reply = get_openai_response(session_id,question, context)

        # Append the assistant's reply so next turn sees it
        chat_history[session_id].append({"role": "assistant", "content": bot_reply})

        formatted_response = format_chatbot_response(bot_reply)
        return {"response": json.dumps(formatted_response)}

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return {
            "response": json.dumps({
                "Description": "We ran into an error, please retry! I can provide information about our generator models, compare different models, or help you find a dealer. What would you like to know?",
                "Links": [
                    '<a href="https://baumalight.com/product/generators/tx-models" target="_blank">TX Models</a>',
                    '<a href="https://baumalight.com/product/generators/kr-models" target="_blank">KR Models</a>',
                    '<a href="https://baumalight.com/product/generators/qc-singlephase" target="_blank">QC Models</a>'
                ],
                "Questions": []
            })
        }

def format_dealer_response(dealers, postal_code):
    dealer_cards = []
    for dealer in dealers:
        dealer_cards.append(
                                 f"<div class='dealer-card'>\n" +
                                 f"  <h6>{dealer['company']}</h6>\n" +
                                 f"  <div class='dealer-details'>\n" +
                                 f"    <div class='detail-row'>\n" +
                                 f"      <span class='label'>Address:</span>\n" +
                                 f"      <span class='value'>{dealer['address']}</span>\n" +
                                 f"    </div>\n" +
                                 f"    <div class='detail-row'>\n" +
                                 f"      <span class='label'>Phone:</span>\n" +
                                 f"      <span class='value'>{dealer['phone']}</span>\n" +
                                 f"    </div>\n" +
                                 f"    <div class='detail-row'>\n" +
                                 f"      <span class='label'>Email:</span>\n" +
                                 f"      <span class='value'>{dealer['email']}</span>\n" +
                                 f"    </div>\n" +
                                 f"  </div>\n" +
                                 f"</div>"
        )
    
    return {
        "Description": f"<div class='dealer-card initial-prompt'>\n" +
                      f"  <div class='detail-row'>\n" +
                      f"    <span class='value'>Here are the dealers closest to {postal_code}:</span>\n" +
                      f"  </div>\n" +
                      f"</div>\n\n" +
                      "\n\n".join(dealer_cards),
        "Links": [
            '<a href="https://baumalight.com/product/locator" target="_blank">Dealer Locator</a>'
        ],
        "Questions": []
    }        
def build_context(chunks):
    """
    Builds a readable context string from retrieved vector chunks.
    Includes model name if present in the chunk metadata.
    """
    context_lines = []
    for chunk in chunks:
        label = f" ({chunk.model})" if hasattr(chunk, "model") and chunk.model else ""
        context_lines.append(f"From {chunk.source}{label}:\n{chunk.content}")
    return "\n\n".join(context_lines)

def clean_response_text(text: str) -> str:
    """Clean up response text by removing source file references and question-answer format."""
    # Remove source file references
    text = re.sub(r'Content from (?:product\d+\.txt|comparisons\.json):\s*', '', text)
    # Remove any duplicate "Content from" lines
    text = re.sub(r'Content from .*?:\s*', '', text)
    # Remove question-answer format
    text = re.sub(r'question":\s*"[^"]*"\s*,\s*"answer":\s*"', '', text)
    # Remove any leading/trailing whitespace and newlines
    text = text.strip()
    return text

def get_openai_response(session_id, question, context):
    # 1) Count questions: how many KR/TX/QC models?
    if re.search(r'how many.*models.*listed', question, re.IGNORECASE):
        q = question.upper()
        if "KR" in q:
            kr = sorted([m for m in model_comparisons if m.startswith("KR")])
            return f"<p>There are {len(kr)} KR models listed: {', '.join(kr)}.</p>"
        if "TX" in q:
            tx = sorted([m for m in model_comparisons if m.startswith("TX")])
            return f"<p>There are {len(tx)} TX models listed: {', '.join(tx)}.</p>"
        if "QC" in q:
            qc = sorted([m for m in model_comparisons if m.startswith("QC")])
            return f"<p>There are {len(qc)} QC models listed: {', '.join(qc)}.</p>"


    # 3) Otherwise fall back to your LLM + vector-context logic
    full_history   = chat_history[session_id]
    recent_history = full_history[-6:]
    system_prompt  = {
        "role": "system",
        "content": """You are a knowledgeable Baumalight product assistant. You will be facing customers, respond accordingly. Always answer with confidence, do not use words such as likely. Recommend one generator model based only on values from the comparisons.json file.DO NOT HALLUCINATE.

Rules: ONLY MODELS THAT EXIST ARE - 

Single Phase (Recommend these first unless the user asks for a three phase generator):
TX7,TX12,TX18,TX25,TX31,KR30,KR44,KR52,KR65,QC12,QC19,QC30,QC45,QC55,QC65,QC80,QC100


The three phase:
QC30-2,QC45-2,QC68-2,QC105-2,QC30-2,QC45-3,QC68-3,QC105-3,QC30-4,QC45-4,QC68-4,QC105-4,QC27-6,QC50-6,QC75-6,QC100-6.

When asked for a recommendation, do not beat around the bush.

Minimum 11hp tractor is needed to run Baumalight Generators (TX7).
Recommend from all TX KR and QC series based on requirement.

DO NOT MENTION ANY OTHER MODEL APART FROM THESE MODELS.

TX series cannot power motors above 2kw.

Always recommend cheaper options and mention it.

If user's tractor Horsepower is above the model's "tractor Horsepower required to run at 100% load", recommend that model

ALWAYS RECOMMEND A MODEL THAT HAS HIGHER KW THAN THE USER REQUIREMENT BECAUSE IT IS NOT RECOMMENDED TO RUN THE GENERATOR AT 100% LOAD AT ALL TIMES.

Suggest TX generator for small loads and brief usage because they are cheap and 2 pole.

Always suggest QC and KR generators for 24/7 output because they are 4 pole.

Suggest QC generators for premium features and three phase outputs.

Model names have the power rating in the name itself. Eg QC12 = 12kW, TX31 = 31kW, QC100 = 100kW.

When the user describes their needs (e.g., "recommend a model for 12KW requirement." or "What should I buy for backup?"):
Do power calculation for their needs and recommend a few similar models.
1. Recommend a KW value slightly above the user requirement and ask them to confirm.
2. After confirmation recommend one model of generator that meet the requirements. 
3. DO NOT include any specifications (KW, HP, Voltage, RPM, Amps, etc.).
4. DO NOT include any values or numbers.
5. Just explain why that model is suitable for the stated use case.
6. When asked to recommend model Calculate the power requirement and suggest one model at a time.
7. When listing price always list USD, CAD, USD discount and CAD discount.
8. If the user asks more than 5 questions, add "You could contact your nearest dealer for further assistance. Just type the word "Dealer"
9. If the user asks more than 5 questions, add "This bot used generative AI, we are constantly working on making it better. Feel free to give feedback for this conversation with a thumbs-up or thumbs-down "

Other rules :
 
1. Do not list source files, URLs, or unknown values.
2. Do not round, guess, or reformat values.
3. Use values only if present in their correct section:
4. details = specs like KW, RPM, etc.
5. description = shipping weight
6.values = pricing
7.Recommend the lowest model that is in the context and meets the requested KW. 
8.Never suggest models with lower KW than requested.
9. Whenever asked for a suggestion, always suggest one model at a time.
10.DO NOT MAKE UP ANY INFORMATION. Check comparisons.json
11. When asked for recommendation, always recommend a model with around 25% higher power rating than the users requirement for safe use.
12. Suggest different models only when asked.
DO not include model, specifications and pricing when not needed.
Include model name and specifications when necessary:
DO not include your own links.

Formatting:

1. Use only: <p>, <strong>, <ul>, <li>
2. No markdown, no <pre>, no code tags
3. No instruction text or file references

"""
    }
    user_message = {
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}"
    }

    messages = [system_prompt] + recent_history + [user_message]
    logger.info("OpenAI API Input:\n" + json.dumps(messages, indent=2))

    response = openai.ChatCompletion.create(
        model=GPT_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=1000
    )

    logger.info("OpenAI API Response:\n" + json.dumps(response, indent=2))
    reply = response.choices[0].message.content.strip()

    # your existing cleanup
    reply = re.sub(r'\{.*?\}', '', reply, flags=re.DOTALL)
    reply = clean_pre_tags(reply)
    reply = re.sub(r'\.{3,}', '', reply)
    reply = re.sub(r'Content from.*?:\n', '', reply, flags=re.DOTALL)
    reply = re.sub(r'Question:.*?$', '', reply, flags=re.DOTALL)
    reply = re.sub(r'<li><strong>.*?:</strong> Not specified</li>', '', reply)
    reply = re.sub(r'<li><strong>.*?:</strong> </li>', '', reply)
    reply = re.sub(r'\s+', ' ', reply).strip()

    return reply

def format_product_specs(specs_text):
    """Format product specifications into a clean HTML structure."""
    # Split the text into sections using ###
    sections = specs_text.split("###")
    
    # Initialize sections dictionary
    formatted_sections = {
        "shipping": [],
        "pricing": [],
        "description": []
    }
    
    # Process each section
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        # Determine section type
        if section.startswith("Pricing:"):
            current_section = "pricing"
            section = section.replace("Pricing:", "").strip()
        elif section.startswith("Additional Information:"):
            current_section = "description"
            section = section.replace("Additional Information:", "").strip()
        else:
            current_section = "shipping"
        
        # Process lines in the section
        for line in section.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Skip lines containing notes
            if "notes:" in line.lower() or "note:" in line.lower():
                continue
                
            if line.startswith("- "):
                # Handle bullet points
                formatted_line = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', line[2:])
                
                if current_section == "shipping":
                    # Handle nested bullet points for shipping
                    if ":" in formatted_line and " - " in formatted_line:
                        main_point, sub_points = formatted_line.split(":", 1)
                        # Skip if this is a notes section
                        if "notes" in main_point.lower():
                            continue
                        formatted_sections["shipping"].append(f"<li>{main_point}:<ul>")
                        for sub_point in sub_points.split(" - "):
                            if sub_point.strip():
                                formatted_sections["shipping"].append(f"<li>{sub_point.strip()}</li>")
                        formatted_sections["shipping"].append("</ul></li>")
                    else:
                        # Skip if this is a notes line
                        if "notes" in formatted_line.lower():
                            continue
                        formatted_sections["shipping"].append(f"<li>{formatted_line}</li>")
                elif current_section == "pricing":
                    # Clean up pricing line
                    formatted_line = formatted_line.replace("Regular Price", "Price")
                    formatted_sections["pricing"].append(f"<li>{formatted_line}</li>")
                elif current_section == "description":
                    # Handle links in description
                    formatted_line = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2" target="_blank">\1</a>', formatted_line)
                    # Skip if this is a notes line
                    if "notes" in formatted_line.lower():
                        continue
                    formatted_sections["description"].append(f"<li>{formatted_line}</li>")
            elif current_section == "description":
                # Handle non-bullet point lines in description
                formatted_line = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2" target="_blank">\1</a>', line)
                # Skip if this is a notes line
                if "notes" in formatted_line.lower():
                    continue
                formatted_sections["description"].append(f"<li>{formatted_line}</li>")
    
    # Build HTML output
    html_parts = ['<div class="product-specs">']
    
    # Add shipping section
    if formatted_sections["shipping"]:
        html_parts.append(f'''
            <div class="shipping-section">
            <br>
                <h6>Specifications & Availability</h6>
                <ul class="shipping-list">
                    {''.join(formatted_sections["shipping"])}
                </ul>
            </div>
        ''')
    
    # Add pricing section
    if formatted_sections["pricing"]:
        html_parts.append(f'''
            <div class="pricing-section">
                <h6>Pricing</h6>
                <ul class="pricing-list">
                    {''.join(formatted_sections["pricing"])}
                </ul>
            </div>
        ''')
    
    html_parts.append('</div>')
    return '\n'.join(html_parts)

def clean_pre_tags(text: str) -> str:
    """Replace pre and code tags with p tags while preserving their content."""
    # Replace pre tags with p tags
    text = re.sub(r'<pre>(.*?)</pre>', r'<p>\1</p>', text, flags=re.DOTALL)
    # Replace code tags with p tags
    text = re.sub(r'<code>(.*?)</code>', r'<p>\1</p>', text, flags=re.DOTALL)
    # Remove any remaining pre or code tags
    text = text.replace('<pre>', '<p>').replace('</pre>', '</p>')
    text = text.replace('<code>', '<p>').replace('</code>', '</p>')
    return text

def format_chatbot_response(response):
    # Parse the response into sections
    sections = {
        "Description": "",
        "Links": [],
        "Questions": []
    }
    
    try:
        # Clean any pre tags from the response first
        response = clean_pre_tags(response)
        
        # Try to parse as JSON first
        if isinstance(response, str) and (response.startswith('{') or response.startswith('[')):
            try:
                data = json.loads(response)
                if isinstance(data, dict):
                    # Format the response as HTML
                    html_parts = []
                    
                    # Add details section if present
                    if 'details' in data:
                        html_parts.append('<div class="specs-section">')
                        for key, value in data['details'].items():
                            if value:  # Only add non-empty values
                                html_parts.append(f'<p><strong>{key}:</strong> {value}</p>')
                        html_parts.append('</div>')
                    
                    # Add description section if present
                    if 'description' in data:
                        html_parts.append('<div class="description-section">')
                        for key, value in data['description'].items():
                            if value and key not in ['notes', 'discountDescription']:  # Skip notes and empty descriptions
                                html_parts.append(f'<p><strong>{key}:</strong> {value}</p>')
                        html_parts.append('</div>')
                    
                    # Add values section if present
                    if 'values' in data:
                        html_parts.append('<div class="pricing-section">')
                        for key, value in data['values'].items():
                            if value and key not in ['leadTime', 'lead time']:  # Skip duplicate lead time
                                html_parts.append(f'<p><strong>{key}:</strong> {value}</p>')
                        html_parts.append('</div>')
                    
                    sections["Description"] = '\n'.join(html_parts)
                else:
                    # Clean up the response
                    clean_response = response.replace('{"', '').replace('"}', '')
                    clean_response = clean_response.replace('category":"', '').replace('qa_pairs":[', '')
                    clean_response = clean_response.replace(']}', '').replace(']', '')
                    clean_response = clean_response.strip()
                    sections["Description"] = f'<p>{clean_response}</p>'
            except json.JSONDecodeError:
                # Clean up the response
                clean_response = response.replace('{"', '').replace('"}', '')
                clean_response = clean_response.replace('category":"', '').replace('qa_pairs":[', '')
                clean_response = clean_response.replace(']}', '').replace(']', '')
                clean_response = clean_response.strip()
                sections["Description"] = f'<p>{clean_response}</p>'
        # Check if the response is from general_answers.json
        elif "Content from general_answers.json" in response:
            try:
                # Extract the JSON content
                json_content = response.split("Content from general_answers.json:\n")[1]
                # Find the first question-answer pair
                first_qa = json_content.split("\n- question:")[1].split("\n  answer:")[1].strip()
                # Clean up the answer
                answer = first_qa.strip('"').strip(',').strip()
                answer = answer.replace('{"', '').replace('"}', '')
                answer = answer.replace('category":"', '').replace('qa_pairs":[', '')
                answer = answer.replace(']}', '').replace(']', '')
                answer = answer.strip()
                
                # Format the response with only the specific answer
                sections["Description"] = f'<p>{answer}</p>'
            except Exception as e:
                logger.error(f"Error formatting general_answers response: {str(e)}")
                # If there's an error, try to extract just the answer part
                try:
                    answer_match = re.search(r'"answer":\s*"([^"]*)"', response)
                    if answer_match:
                        answer = answer_match.group(1)
                        # Clean up the answer
                        answer = answer.replace('{"', '').replace('"}', '')
                        answer = answer.replace('category":"', '').replace('qa_pairs":[', '')
                        answer = answer.replace(']}', '').replace(']', '')
                        answer = answer.strip()
                        sections["Description"] = f'<p>{answer}</p>'
                    else:
                        # Clean up the response
                        clean_response = response.replace('{"', '').replace('"}', '')
                        clean_response = clean_response.replace('category":"', '').replace('qa_pairs":[', '')
                        clean_response = clean_response.replace(']}', '').replace(']', '')
                        clean_response = clean_response.strip()
                        sections["Description"] = f'<p>{clean_response}</p>'
                except:
                    # Clean up the response
                    clean_response = response.replace('{"', '').replace('"}', '')
                    clean_response = clean_response.replace('category":"', '').replace('qa_pairs":[', '')
                    clean_response = clean_response.replace(']}', '').replace(']', '')
                    clean_response = clean_response.strip()
                    sections["Description"] = f'<p>{clean_response}</p>'
        else:
            # For all other responses, use the model's content directly
            # Clean up the response
            clean_response = response.replace('{"', '').replace('"}', '')
            clean_response = clean_response.replace('category":"', '').replace('qa_pairs":[', '')
            clean_response = clean_response.replace(']}', '').replace(']', '')
            clean_response = clean_response.strip()
            sections["Description"] = f'<p>{clean_response}</p>'
    except Exception as e:
        logger.error(f"Error in format_chatbot_response: {str(e)}")
        # Clean up the response
        clean_response = response.replace('{"', '').replace('"}', '')
        clean_response = clean_response.replace('category":"', '').replace('qa_pairs":[', '')
        clean_response = clean_response.replace(']}', '').replace(']', '')
        clean_response = clean_response.strip()
        sections["Description"] = f'<p>{clean_response}</p>'
    
    # Add default links if none were found
    if not sections["Links"]:
        sections["Links"] = [
            '<a href="https://baumalight.com/product/generators/tx-models" target="_blank">TX Models</a>',
            '<a href="https://baumalight.com/product/generators/kr-models" target="_blank">KR Models</a>',
            '<a href="https://baumalight.com/product/generators/qc-singlephase" target="_blank">QC Models</a>'
        ]

    return sections

@app.post("/rebuild-index")
async def rebuild_index():
    try:
        build_vector_index()
        return {
            "status": "success",
            "message": "Index rebuilt"
        }
    except Exception as e:
        logger.error(f"Failed to rebuild index: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

def generate_model_comparison(models_to_compare):
    """Generate a comparison between two models."""
    comparison_data = {"models": {}}
    
    # Find the models in the comparison data
    model_series = {}  # Track which series each model belongs to
    for model_name in models_to_compare:
        # Clean up model name for matching
        clean_model_name = model_name.upper().replace(" ", "").replace("-", "")
        prefix = re.match(r'([A-Za-z]+)', model_name).group(1)
        
        # Search for this model in all series
        found = False
        for series in model_comparisons:
            for model_key, model_data in model_comparisons[series].items():
                # Clean up the model key for matching
                clean_key = model_key.upper().replace(" ", "").replace("-", "")
                # Match either exact or as a prefix (for phase variations)
                if clean_key == clean_model_name or clean_model_name in clean_key:
                    comparison_data["models"][model_key] = model_data
                    # Store which series this model belongs to
                    if prefix == "QC":
                        model_series[model_key] = "qc-singlephase"
                    elif prefix == "TX":
                        model_series[model_key] = "tx-series"
                    elif prefix == "KR":
                        model_series[model_key] = "kr-models"
                    found = True
                    break
            if found:
                break
    
    # If we found at least 2 models, create a comparison
    if len(comparison_data["models"]) >= 2:
        model_list = list(comparison_data["models"].items())
        model1_name, model1_data = model_list[0]
        model2_name, model2_data = model_list[1]
        
        # Get KW values for comparison
        model1_kw = model1_data['details'].get('KW', 0)
        model2_kw = model2_data['details'].get('KW', 0)
        
        # Determine which model has higher power output
        power_comparison = ""
        if model1_kw > model2_kw:
            power_comparison = f"<p class='power-comparison'><strong>Power Comparison:</strong> The {model1_name} offers higher power output ({model1_kw} KW vs {model2_kw} KW), making it better suited for more demanding applications.</p>"
        elif model2_kw > model1_kw:
            power_comparison = f"<p class='power-comparison'><strong>Power Comparison:</strong> The {model2_name} offers higher power output ({model2_kw} KW vs {model1_kw} KW), making it better suited for more demanding applications.</p>"
        else:
            power_comparison = f"<p class='power-comparison'><strong>Power Comparison:</strong> Both models have the same power output of {model1_kw} KW.</p>"
        
        comparison_html = f"""<div class='comparison-section'>
            {power_comparison}
            <div class='model-card'>
            <br>
                <p><strong>{model1_name}</strong></p>
                <div class='specs-section'>
                    <p><strong>Power Output:</strong> {model1_data['details'].get('KW', 'N/A')} KW</p>
                    <p><strong>Surge Capacity:</strong> {model1_data['details'].get('Momentary Surge KW', 'N/A')} KW</p>
                    <p><strong>Load Capacity:</strong> {model1_data['details'].get('50% Load (HP)', 'N/A')} HP at 50% load | {model1_data['details'].get('100% Load (HP)', 'N/A')} HP at 100% load</p>
                    <p><strong>Voltage/Phase:</strong> {model1_data['details'].get('Volts', 'N/A')} - {model1_data['details'].get('Phase', 'N/A')}-phase</p>
                    <p><strong>Internal RPM:</strong> {model1_data['details'].get('Internal RPM', 'N/A')}</p>
                    <p><strong>Full Output Amps:</strong> {model1_data['details'].get('Full Output Amps (Main Breaker)', 'N/A')}</p>
                </div>
                <div class='pricing-section'>
                    <p><strong>USD Price:</strong> ${model1_data['values'].get('USDPrice', 'N/A')}</p>
                    <p><strong>USD Discount:</strong> ${model1_data['values'].get('discountUSDPrice', 'N/A')}</p>
                    <p><strong>CAD Price:</strong> ${model1_data['values'].get('CADPrice', 'N/A')}</p>
                    <p><strong>CAD Discount:</strong> ${model1_data['values'].get('discountCADPrice', 'N/A')}</p>
                    <p><strong>Lead Time:</strong> {model1_data['values'].get('leadTime', model1_data['values'].get('lead time', 'N/A'))}</p>
                </div>
            </div>

            <div class='model-card'>
                <p><strong>{model2_name}</strong></p>
                <div class='specs-section'>
                    <p><strong>Power Output:</strong> {model2_data['details'].get('KW', 'N/A')} KW</p>
                    <p><strong>Surge Capacity:</strong> {model2_data['details'].get('Momentary Surge KW', 'N/A')} KW</p>
                    <p><strong>Load Capacity:</strong> {model2_data['details'].get('50% Load (HP)', 'N/A')} HP at 50% load | {model2_data['details'].get('100% Load (HP)', 'N/A')} HP at 100% load</p>
                    <p><strong>Voltage/Phase:</strong> {model2_data['details'].get('Volts', 'N/A')} - {model2_data['details'].get('Phase', 'N/A')}-phase</p>
                    <p><strong>Internal RPM:</strong> {model2_data['details'].get('Internal RPM', 'N/A')}</p>
                    <p><strong>Full Output Amps:</strong> {model2_data['details'].get('Full Output Amps (Main Breaker)', 'N/A')}</p>
                </div>
                <div class='pricing-section'>
                    <p><strong>USD Price:</strong> ${model2_data['values'].get('USDPrice', 'N/A')}</p>
                    <p><strong>USD Discount:</strong> ${model2_data['values'].get('discountUSDPrice', 'N/A')}</p>
                    <p><strong>CAD Price:</strong> ${model2_data['values'].get('CADPrice', 'N/A')}</p>
                    <p><strong>CAD Discount:</strong> ${model2_data['values'].get('discountCADPrice', 'N/A')}</p>
                    <p><strong>Lead Time:</strong> {model2_data['values'].get('leadTime', model2_data['values'].get('lead time', 'N/A'))}</p>
                </div>
            </div>
        </div>"""
        
       
    
    return None


if __name__ == "__main__":

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
