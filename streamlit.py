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

os.environ['HF_HOME'] = '/path/to/your/hf/cache'
os.environ['HUGGINGFACE_TOKEN'] = 
API_KEY_COHERE = 
co = cohere.Client(API_KEY_COHERE)

st.title("المساعد القانوني الذكي")

assert torch.cuda.is_available(), "GPU is not available. Please check your runtime settings."
device = "cuda"

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

def main():
    global tokenizer, model, classifier, bm25 

    assert torch.cuda.is_available(), "GPU is not available. Please check your runtime settings."
    
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

    with open('madany.txt', 'r', encoding='utf-8') as file:
        extracted_text = file.read()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=500)
    if 'doc_splits' not in st.session_state: 
        st.session_state.doc_splits = text_splitter.create_documents([extracted_text])
    
    doc_splits = st.session_state.doc_splits

    model = st.session_state.model
    classifier = st.session_state.classifier
    tokenizer = st.session_state.tokenizer

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

def google_search(state: GraphState, number_of_results=10) -> GraphState:
    print("---Google Search---")
    question = state["question"]
    print(f"Searching for: {question}")

    google_results = []
    state["documents"] = []

    try:
        google_results = list(search(question, number_of_results))
    except Exception as e:
        print(f"Error during Google search: {e}")

    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})

    if google_results:
        for url in google_results:
            print(f"Trying URL: {url}")
            try:
                response = session.get(url, verify=True)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')

                paragraphs = soup.find_all('p')
                page_content = "\n".join(paragraph.get_text() for paragraph in paragraphs)

                document = Document(page_content=page_content)
                state.setdefault("documents", []).append(document)
                print("Extracted content from the URL.")
                break
            except requests.exceptions.HTTPError as http_error:
                if http_error.response.status_code == 403:
                    print(f"Access forbidden for {url}. Trying the next URL...")
                    continue
                else:
                    print(f"HTTP Error for {url}: {http_error}")
                    break
            except requests.exceptions.SSLError as ssl_error:
                print(f"SSL Error for {url}: {ssl_error}")
                continue
            except Exception as e:
                print(f"Error fetching or parsing the URL: {e}")
                continue
    else:
        print("No Google results found.")

    return state

def clean_text(generation):
    answer_start = re.split(r'(الإجابة\s*[:.]?)', generation, maxsplit=1, flags=re.IGNORECASE)
    if len(answer_start) > 2:
        cleaned = answer_start[-1]
    else:
        cleaned = generation

    patterns_to_remove = [
        r'^.*?(الإجابة\s*[:.]?)',
        r'الجواب\s*:.*?(?=الإجابة|$)',
        r'المطلوب\s*:.*?(?=الإجابة|$)',
        r'ملخص\s*:.*?(?=الإجابة|$)',
        r'التعليمات.*?(?=الإجابة|$)',
    ]
    
    combined_pattern = re.compile(
        r'(' + '|'.join(patterns_to_remove) + r')',
        flags=re.IGNORECASE | re.UNICODE | re.DOTALL
    )
    
    cleaned = combined_pattern.sub('', cleaned)
    
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'\s*([\.،:;])\s*', r'\1 ', cleaned)
    cleaned = re.sub(r'^\W+|\W+$', '', cleaned)

    if not re.search(r'[\u0600-\u06FF]', cleaned):
        return "لا تتوفر معلومات كافية"
    
    cleaned = re.sub(r'(\d+)\.', r'\1-', cleaned)
    return cleaned.strip()
    
def summarize_document(state: GraphState) -> GraphState:
    if "Messages" not in state:
        state["Messages"] = []

    if "documents" in state and state["documents"]:
        document = state["documents"][0]
        if isinstance(document, Document):
            try:
                content = document.page_content[:3000]
                cleaned_content = re.sub(r'\s+', ' ', content).strip()

                model_input = f"""
                النص القانوني:
                {cleaned_content}

                التعليمات:
                1. لخص المحتوى في 3-5 جمل قصيرة
                2. ركز على العناصر القانونية الأساسية
                3. تجنب التفاصيل الثانوية
                4. استخدم لغة عربية فصيحة
                5. لا تقوم بالاجابه باى لغه سوى العربيه
                6. لا تقوم باضافه هذه التعليمات فى اجاباتك
                الملخص:
                """

                inputs = st.session_state.tokenizer(
                    model_input,
                    return_tensors="pt",
                    max_length=4096,
                    truncation=True
                ).to(model.device)

                with torch.no_grad():
                    response = st.session_state.model.generate(
                        **inputs,
                        max_new_tokens=200,
                        no_repeat_ngram_size=2,
                        do_sample=True,
                        temperature=0.5
                    )

                raw_summary = st.session_state.tokenizer.decode(response[0], skip_special_tokens=True)
                cleaned_summary = clean_text(raw_summary)

                summary_lines = [line.strip() for line in cleaned_summary.split('.') if line.strip()][:3]
                final_summary = '. '.join(summary_lines)

                state["Messages"].append(final_summary)

            except Exception as e:
                print(f"Summarization error: {e}")
                state["summary"] = "Error in summarization process"
        else:
            state["summary"] = "Error: Invalid document format"
    else:
        state["summary"] = "No documents available to summarize."

    return state

def call_llm_with_top_paragraphs(state: GraphState) -> GraphState:
    question = clean_text(state.get("question", ""))
    context = state.get("Messages", ["لا يوجد سياق"])[-1][:500]

    prompt_template = """
    السؤال القانوني:
    {question}

    السياق المرجعي:
    {context}

    التعليمات الصارمة:
    1. ابدأ الإجابة مباشرة بعبارة "الإجابة هي:"
    2. قدم النقاط القانونية الرئيسية فقط
    3. استخدم الترقيم العربي (١، ٢، ٣)
    4. تجنب أي ذكر للتعليمات أو السياق
    5. أجب حصرياً باللغة العربية الفصحى

    الإجابة:
    """.strip()

    inputs = st.session_state.tokenizer(
        prompt_template.format(question=question, context=context),
        return_tensors="pt",
        max_length=1024,
        truncation=True
    ).to(model.device)

    with torch.no_grad():
        response = st.session_state.model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            top_k=40,
            top_p=0.9,
            temperature=0.4,
            repetition_penalty=1.2
        )

    raw_answer = st.session_state.tokenizer.decode(response[0], skip_special_tokens=True)
    cleaned_answer = clean_text(raw_answer)
    
    if not cleaned_answer.startswith("الإجابة"):
        cleaned_answer = f"الإجابة هي:\n{cleaned_answer}"
    
    cleaned_answer = re.sub(r'(\d+)-', lambda m: f"{int(m.group(1))}٫", cleaned_answer)
    
    state["generation"] = cleaned_answer
    state["Messages"].append(f"Q: {question}\nA: {state['generation']}")
    return state

workflow.add_node("retrieve", retrieve)
workflow.add_node("google_search", google_search)
workflow.add_node("summarize_document", summarize_document)
workflow.add_node("call_llm", call_llm_with_top_paragraphs)

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
        self.origin = origin
        self.message = message

def load_css():
    st.markdown("""
    <style>
    :root {
        --primary-color: #2c3e50;
        --secondary-color: #3498db;
        --gold-accent: #d4af37;
        --text-color: #2c3e50;
    }
    
    .chat-row {
        display: flex;
        margin: 1.5rem 0;
        direction: rtl;
    }
    
    .row-reverse {
        flex-direction: row-reverse;
    }
    
    .chat-bubble {
        padding: 1.2rem 1.8rem;
        border-radius: 25px;
        max-width: 75%;
        font-family: 'Noto Sans Arabic', Tahoma, sans-serif;
        font-size: 1.1rem;
        line-height: 1.8;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        position: relative;
    }
    
    .human-bubble {
        background: var(--primary-color);
        color: white;
        margin-left: 10%;
        border-bottom-right-radius: 5px;
    }
    
    .ai-bubble {
        background: #f8f9fa;
        color: var(--text-color);
        border: 2px solid var(--gold-accent);
        margin-right: 10%;
        border-bottom-left-radius: 5px;
    }
    
    .stTextInput>div>div>input {
        text-align: right;
        padding: 15px;
        font-size: 1.1rem;
        border: 2px solid var(--gold-accent);
        border-radius: 8px;
    }
    
    .stButton>button {
        background: var(--gold-accent);
        color: var(--primary-color);
        font-weight: bold;
        padding: 12px 30px;
        border-radius: 8px;
        border: none;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: #b39530;
        transform: scale(1.05);
    }
    
    @font-face {
        font-family: 'Noto Sans Arabic';
        font-style: normal;
        font-weight: 400;
        src: url(https://fonts.gstatic.com/s/notosansarabic/v18/nwpxtLGrOAZMl5nJ_wfgRg3DrWFZWsnVBJ_sS6tlqHHFlj4wv4o.woff2) format('woff2');
    }
    
    .stMarkdown h1 {
        text-align: right;
        color: var(--primary-color);
        border-bottom: 3px solid var(--gold-accent);
        padding-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if 'conversation' not in st.session_state:
        st.session_state.conversation = ConversationChain(
            llm=llm,
            memory=ConversationSummaryMemory(llm=llm),
        )

def on_click_callback():
    human_prompt = st.session_state.human_prompt
    st.session_state.history.append(Message("human", human_prompt))

    state = {
        "question": human_prompt,
        "generation": "",
        "documents": [],
        "Messages": []
    }

    with st.spinner("جارٍ معالجة طلبك..."):
        try:
            final_state = st.session_state.app.invoke(state)
            if final_state["generation"]:
                cleaned_response = clean_text(final_state["generation"])
                st.session_state.history.append(Message("ai", cleaned_response))
        except Exception as e:
            st.error(f"حدث خطأ في المعالجة: {str(e)}")
            st.session_state.history.append(Message("ai", "حدث خطأ غير متوقع. يرجى المحاولة لاحقًا."))

load_css()
initialize_session_state()

chat_placeholder = st.container()
prompt_placeholder = st.form("chat-form")

with chat_placeholder:
    for chat in st.session_state.history:
        div = f"""
        <div class="chat-row {'row-reverse' if chat.origin == 'human' else ''}">
            <div class="chat-bubble {'ai-bubble' if chat.origin == 'ai' else 'human-bubble'}">
                &#8203;{chat.message}
            </div>
        </div>
        """
        st.markdown(div, unsafe_allow_html=True)

with prompt_placeholder:
    cols = st.columns((6, 1))
    cols[0].text_input(
        "أدخل استفسارك القانوني هنا",
        value="",
        label_visibility="collapsed",
        key="human_prompt",
    )
    cols[1].form_submit_button(
        "إرسال", 
        type="primary", 
        on_click=on_click_callback, 
    )

components.html("""
<script>
document.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        document.querySelector('button[type="submit"]').click();
    }
});
</script>
""", height=0, width=0)
