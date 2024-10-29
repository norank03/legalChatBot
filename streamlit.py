import streamlit as st
import os
import numpy as np
import pandas as pd
import cohere
import torch
import faiss
import re
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain import LLMChain, PromptTemplate
from langchain.llms import HuggingFacePipeline
from googlesearch import search
from typing import List
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from pprint import pprint
import requests
from bs4 import BeautifulSoup
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import LLMChain, PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
import streamlit.components.v1 as components
# Environment Variables
os.environ['HF_HOME'] = '/path/to/your/hf/cache'
os.environ['HUGGINGFACE_TOKEN'] = ''  
API_KEY_COHERE = ''
co = cohere.Client(API_KEY_COHERE)

# Streamlit chatbot interface
st.title("Legal Chatbot")

# Load the tokenizer and model
assert torch.cuda.is_available(), "GPU is not available. Please check your runtime settings."
device = "cuda"



# Document class
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

# Main chatbot flow
def main():
    global tokenizer, model, classifier, bm25 

    # Ensure you're using the GPU
    assert torch.cuda.is_available(), "GPU is not available. Please check your runtime settings."
    
    # Initialize tokenizer, model, and classifier
    if 'tokenizer' not in st.session_state:
        st.session_state.tokenizer = AutoTokenizer.from_pretrained("Nourankoro/Last_Jais")

    if 'model' not in st.session_state:
        st.session_state.model = AutoModelForCausalLM.from_pretrained(
            "Nourankoro/Last_Jais",
            load_in_4bit=True,
            device_map="auto"
        )

    if 'classifier' not in st.session_state:
        st.session_state.classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")

    # Load document
    with open('madany.txt', 'r', encoding='utf-8') as file:
        extracted_text = file.read()

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=500)
    if 'doc_splits' not in st.session_state: 
        st.session_state.doc_splits = text_splitter.create_documents([extracted_text])
    
    doc_splits = st.session_state.doc_splits

    model = st.session_state.model
    classifier = st.session_state.classifier
    tokenizer = st.session_state.tokenizer

    # Initialize BM25
    tokenized_corpus = [bm25_tokenizer(passage.page_content) for passage in tqdm(doc_splits)]
    if 'bm25' not in st.session_state:
        st.session_state.bm25 = BM25Okapi(tokenized_corpus)
    
    bm25 = st.session_state.bm25 


def bm25_tokenizer(text):
    return re.findall(r'\w+', text.lower())

def keyword_search(query, top_k=3, num_candidates=15):
    bm25_scores = bm25.get_scores(bm25_tokenizer(query))
    top_n = np.argpartition(bm25_scores, -num_candidates)[-num_candidates:]
    bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
    bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)

    bm25_results = []
    for hit in bm25_hits[:top_k]:
        document_text = st.session_state.doc_splits[hit['corpus_id']].page_content
        bm25_results.append(Document(page_content=document_text.replace("\n", " "),
                                      metadata={'corpus_id': hit['corpus_id'], 'score': hit['score']}))

    if not bm25_results:
        bm25_results.append(Document(page_content="No relevant documents found."))

    return bm25_results

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]
    Messages: List[Document]

workflow = StateGraph(GraphState)

def route_question(state: dict) -> str:
    """Route a user question to either a vector store or Google Search using an LLM."""
    print("---Route Question---")
    
    question = state["question"]
    candidate_labels = ["legal", "non-legal"]
    result = st.session_state.classifier(question, candidate_labels)
    
    legal_score = result['scores'][0]
    non_legal_score = result['scores'][1]

    if legal_score > 0.60:
        print("Routing to retrieve (legal question)...")
        llm_response = "retrieve"
    else:
        print("Routing to Google search (non-legal question)...")
        llm_response = "google_search"
    
    return llm_response

def retrieve(state: GraphState) -> GraphState:
    """Retrieve documents using unified search based on the model."""
    print("---Retrieve Documents---")

    question = state["question"]
    if "documents" not in state:
        state["documents"] = []

    documents = keyword_search(question)

    if documents:
        state["documents"].extend(documents)
        print(f"Documents retrieved: {[doc.page_content for doc in documents]}")
    else:
        print("No documents found in State Retrieve.")
        state["documents"].append(Document("No relevant documents found."))

    print(f"All documents in state Retrieve: {[doc.page_content for doc in state['documents']]}")
    return state

def RetrievalGrader(state):
    """Grades the relevance of a retrieved document based on a user question."""
    
    question = state.get("question", "")
    docs = state.get("documents", [])
    
    if not docs:
        print("No documents found for grading.")
        return "google_search"
    
    doc_txt = docs[0].page_content
    question_embedding = co.embed(texts=[question]).embeddings[0]
    doc_embedding = co.embed(texts=[doc_txt]).embeddings[0]

    question_tensor = torch.tensor(question_embedding)
    doc_tensor = torch.tensor(doc_embedding)

    similarity_score = torch.nn.functional.cosine_similarity(question_tensor.unsqueeze(0), doc_tensor.unsqueeze(0))
    relevance_score = torch.sigmoid(similarity_score).item()

    print(f"Relevance Score: {relevance_score}")

    if relevance_score < 0.75:
        return "google_search"
    else:
        return "summarize_document"



from googlesearch import search  # Ensure this library is properly installed

def google_search(state: GraphState, number_of_results=10) -> GraphState:
    """Google search based on the question and retrieve the first successful result."""
    print("---Google Search---")
    question = state["question"]
    print(f"Searching for: {question}")

    google_results = []
    state["documents"] = []

    try:
        # Use correct argument for number of results
        google_results = list(search(question, number_of_results))
    except Exception as e:
        print(f"Error during Google search: {e}")

    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})

    if google_results:
        for url in google_results:
            print(f"Trying URL: {url}")
            try:
                response = session.get(url, verify=True)  # Change to True for SSL verification
                response.raise_for_status()  # Raise an error for bad responses
                soup = BeautifulSoup(response.content, 'html.parser')

                paragraphs = soup.find_all('p')
                page_content = "\n".join(paragraph.get_text() for paragraph in paragraphs)

                document = Document(page_content=page_content)
                state.setdefault("documents", []).append(document)
                print("Extracted content from the URL.")
                break  # Exit loop if successful
            except requests.exceptions.HTTPError as http_error:
                if http_error.response.status_code == 403:
                    print(f"Access forbidden for {url}. Trying the next URL...")
                    continue  # Try the next URL
                else:
                    print(f"HTTP Error for {url}: {http_error}")
                    break  # Exit loop for other HTTP errors
            except requests.exceptions.SSLError as ssl_error:
                print(f"SSL Error for {url}: {ssl_error}")
                continue  # Continue to the next URL
            except Exception as e:
                print(f"Error fetching or parsing the URL: {e}")
                continue  # Continue to the next URL
    else:
        print("No Google results found.")

    return state





def summarize_document(state: GraphState) -> GraphState:
    """Summarize the retrieved document using the LLM."""
    if "Messages" not in state:
        state["Messages"] = []

    if "documents" in state and state["documents"]:
        document = state["documents"][0]
        if isinstance(document, Document):
            content = document.page_content
            user_input = state.get("question", "")
            combined_content = "\n\n".join(state["Messages"] + [content, user_input])

            # Create a prompt for summarization
            model_input = (
                f"Summarize the following document in 400 words or less, ensuring the summary is coherent and logical:\n\n{content}\n\n"
                "Only provide the summary without repeating the document or sentences, words , or the question."
            )

            # Tokenize input and handle input length
            inputs = st.session_state.tokenizer(model_input, return_tensors="pt").to(model.device)

            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']

            # Ensure the input does not exceed the model's maximum input length
            max_input_length = st.session_state.model.config.max_position_embeddings
            if inputs['input_ids'].shape[1] > max_input_length:
                inputs['input_ids'] = inputs['input_ids'][:, :max_input_length]
                if 'attention_mask' in inputs:
                    inputs['attention_mask'] = inputs['attention_mask'][:, :max_input_length]

            # Generate a summary
            with torch.no_grad():
                response = st.session_state.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    no_repeat_ngram_size=2,
                    do_sample=True,
                    early_stopping=True
                )

            # Decode the generated summary
            summary = st.session_state.tokenizer.decode(response[0], skip_special_tokens=True).strip()

            # Check if the summary exceeds the desired length
            if len(summary.split()) > 50:
                summary = ' '.join(summary.split()[:50])  # Trim to first 50 words

            state["Messages"].append(summary)
        else:
            state["summary"] = "Error: The first document is not valid."
    else:
        state["summary"] = "No documents available to summarize."

    return state




def clean_text(generation):
    # Define the sentences to be removed
    sentences_to_remove = {
        "أعطِ إجابة واضحة ومباشرة، تعتمد فقط على المعلومات الموجودة في السياق.": '',
        "تجنب تكرار أي جزء من السؤال أو السياق في الإجابة.": ''
    }

    # Create a regex pattern that matches any of the sentences
    pattern = re.compile('|'.join(re.escape(sentence) for sentence in sentences_to_remove.keys()))

    # Use the pattern to substitute the sentences with an empty string
    cleaned_generation = pattern.sub(lambda match: sentences_to_remove[match.group(0)], generation)

    # Clean up any extra whitespace left after removal
    cleaned_generation = re.sub(r'\s+', ' ', cleaned_generation).strip()

    return cleaned_generation

def call_llm_with_top_paragraphs(state: GraphState) -> GraphState:
    """Call LLM using the summarized document content."""
    
    question = state.get("question")
    latest_message = state.get("Messages", ["No relevant summary available."])[-1]

    # Clean the latest message
    latest_message = clean_text(latest_message)
    
    if "generation" not in state:
        state["generation"] = ""

    model_input = (
        f"السؤال: {question}\n"
        f"السياق: {latest_message}\n\n"
        "الجواب: أعطِ إجابة واضحة ومباشرة، تعتمد فقط على المعلومات الموجودة في السياق. "
        "تجنب تكرار أي جزء من السؤال أو السياق في الإجابة."
    )

    inputs = st.session_state.tokenizer(model_input, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        response = st.session_state.model.generate(
            **inputs,
            max_new_tokens=260,
            do_sample=True,
            top_k=60,
            top_p=0.96,
            no_repeat_ngram_size=2,
            temperature=0.4
        )

    print("Response IDs:", response)
    
    generation = st.session_state.tokenizer.decode(response[0], skip_special_tokens=True).strip()
    
    # Clean the generated answer
    generation = re.sub(
        r'أعطِ إجابة واضحة ومباشرة، تعتمد فقط على المعلومات الموجودة في السياق\. تجنب تكرار أي جزء من السؤال أو السياق في الإجابة\.',
        '', 
        generation
    )

    # Get embeddings for the question and generated answer
    question_tensor = st.session_state.tokenizer(question, return_tensors="pt").to(model.device)
    generated_answer_tensor = st.session_state.tokenizer(generation, return_tensors="pt").to(model.device)

    # Generate logits instead of trying to get last_hidden_state
    with torch.no_grad():
        question_logits = st.session_state.model(**question_tensor).logits
        generated_answer_logits = st.session_state.model(**generated_answer_tensor).logits

    # Ensure that you are accessing the correct dimension
    # Using the last token's logits for comparison
    question_embedding = question_logits[:, -1, :] if question_logits.dim() > 1 else question_logits
    generated_answer_embedding = generated_answer_logits[:, -1, :] if generated_answer_logits.dim() > 1 else generated_answer_logits

    # Reshape if needed to ensure they are 2D
    if question_embedding.dim() == 1:
        question_embedding = question_embedding.unsqueeze(0)  # Add batch dimension
    if generated_answer_embedding.dim() == 1:
        generated_answer_embedding = generated_answer_embedding.unsqueeze(0)  # Add batch dimension

    # Calculate similarity score
    similarity_score = torch.nn.functional.cosine_similarity(question_embedding, generated_answer_embedding)
    relevance_score = torch.sigmoid(similarity_score).item()  # Get the relevance score

    if relevance_score < 0.60:  # Check if the score is less than 80%
        print("Relevance score is less than 60%, regenerating answer...")
        print(f"relevence score: ",relevance_score)
        return call_llm_with_top_paragraphs(state)  # Recursively call to regenerate answer

    # Clean the generated answer
    generation = clean_text(generation)

    if "Messages" not in state:
        state["Messages"] = []
    
    state["generation"] = generation
    state["documents"] = []
    state["Messages"].append(f"Q: {question}\nA: {state['generation']}")
    print(f"Answer: {state.get('generation')}")

    print("Cleared documents from the state.")

    return state








workflow.add_node("retrieve", retrieve)
workflow.add_node("google_search", google_search)
workflow.add_node("summarize_document", summarize_document)
workflow.add_node("call_llm", call_llm_with_top_paragraphs)

# Define edges based on conditions
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "retrieve": "retrieve",
        "google_search": "google_search",
    },
)

workflow.add_conditional_edges( 
    "retrieve",
    RetrievalGrader,
    {
        "summarize_document": "summarize_document",
        "google_search": "google_search",
    },
)

workflow.add_edge("google_search", "summarize_document") 
workflow.add_edge("summarize_document","call_llm")
workflow.add_edge("call_llm", END)

app = workflow.compile()

if 'app' not in st.session_state:
    st.session_state.app = workflow.compile()

main()

# Load your model pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=st.session_state.tokenizer,
    max_length=1000,
    max_new_tokens=200
)

llm = HuggingFacePipeline(pipeline=pipe)

class Message:
    def __init__(self, origin, message):
        self.origin = origin  # 'human' or 'ai'
        self.message = message

def load_css():
    with open("style.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []

    if 'conversation' not in st.session_state:
        st.session_state.conversation = ConversationChain(
            llm=llm,
            memory=ConversationSummaryMemory(llm=llm),
        )


import re

def on_click_callback():
    # Retrieve user input from session state
    human_prompt = st.session_state.human_prompt

    # Append user input to history
    st.session_state.history.append(Message("human", human_prompt))

    # Initialize the state with user input
    state = {
        "question": human_prompt,
        "generation": "",  # Initialize as an empty string
        "documents": [],   # Initialize as an empty list
        "Messages": []     # Initialize as an empty list
    }

    # Use Streamlit's spinner for UI feedback
    with st.spinner("Generating response..."):
        # Invoke the app and pass the state
        final_state = st.session_state.app.invoke(state)

        # Check if the 'Messages' is present in the final state and has content
        if "generation" in final_state and final_state["generation"]:
            llm_response = final_state["generation"]  # Access the last message directly
            
            # Use regex to remove any English letters or words
            llm_response = re.sub(r'[A-Za-z0-9]+', '', llm_response)

            #st.write("Answer:", llm_response.strip())  # Display the answer without leading/trailing spaces
            st.session_state.history.append(Message("ai", llm_response))  # Append to history
        else:
            st.write("Answer:", "No response generated.")

def load_css():
    with open("style.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)




load_css()
initialize_session_state()

# Load your logo
logo_path = "logo.png"  # Update with your logo path
st.image(logo_path, use_column_width=False, output_format="auto")

# Create placeholders for chat and input form
chat_placeholder = st.container()
prompt_placeholder = st.form("chat-form")

with chat_placeholder:
    # Display chat history
    for chat in st.session_state.history:
        div = f"""
        <div class="chat-row {'row-reverse' if chat.origin == 'human' else ''}">
            <div class="chat-bubble {'ai-bubble' if chat.origin == 'ai' else 'human-bubble'}">
                &#8203;{chat.message}
            </div>
        </div>
        """
        st.markdown(div, unsafe_allow_html=True)

    # Add spacing
    for _ in range(3):
        st.markdown("")

with prompt_placeholder:
    st.markdown("**Chat**")
    cols = st.columns((6, 1))
    cols[0].text_input(
        "Chat",
        value="",
        label_visibility="collapsed",
        key="human_prompt",
    )
    cols[1].form_submit_button(
        "Submit", 
        type="primary", 
        on_click=on_click_callback, 
    )

# JavaScript to handle 'Enter' key submission
components.html("""
<script>
const streamlitDoc = window.parent.document;

const buttons = Array.from(
    streamlitDoc.querySelectorAll('.stButton > button')
);
const submitButton = buttons.find(
    el => el.innerText === 'Submit'
);

streamlitDoc.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') {
        submitButton.click();
    }
});
</script>
""", 
    height=0,
    width=0,
)
