from dotenv import load_dotenv
import requests
from langchain.tools import tool
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_community.chat_message_histories.upstash_redis import UpstashRedisChatMessageHistory
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.vectorstores.base import VectorStore
load_dotenv()
from typing import List
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import csv
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

def read_csv(file_path: str, csv_args: dict = {'delimiter': ','}):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file, **csv_args)
        for row in csv_reader:
            data.append(' '.join(row))
    return data

def chroma_db_tool() -> VectorStore:
    raw_documents = read_csv('assistantweb/data/products.csv/')
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(raw_documents)
    embedding_function = OpenAIEmbeddings()
    db = Chroma.from_documents(chunks, embedding_function)
    return db


@tool
class SearchTool:
    """Search data in ChromaDB using a given query."""
    name = "chroma_db_search"

    def __call__(self, query: str) -> str:
        """Search data in ChromaDB using a given query."""
        db = chroma_db_tool()
        docs = db.similarity_search(query)
        return docs[0].page_content if docs else "No results found."

tools = [
    Tool(
        name="chroma_db_search",
        func=lambda tool_input: SearchTool().__call__(tool_input),
        description="Useful for retrieving data from ChromaDB based on user queries"
    )       
]

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.6
)

prompt = ChatPromptTemplate.from_messages([
        ("system", "eres belgrano autos, un asistente de una tienda de autos con el objetivo de ayudar a los clientes en sus dudas de una manera amable"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])


UPSTASH_URL = "https://us1-excited-haddock-41233.upstash.io"
UPSTASH_TOKEN ="AaERACQgMDliYjdlZWYtYzgzYi00NmJiLTk2MjktNTllNjZjZDBjZWYxMmE5OGRiNGRmZTgxNDc2YjkyMmQ1YzExOWE0NTljNjI="
history = UpstashRedisChatMessageHistory(
    url=UPSTASH_URL, token=UPSTASH_TOKEN,ttl=600, session_id="chat1"
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    chat_memory=history,
)

chain = prompt | model

chain = LLMChain(
    llm=model,
    prompt=prompt,
    verbose=True,
    memory=memory,
    )
_call_=lambda input: SearchTool().run(input, tools),


@csrf_exempt
def ask(request):
    try:
        if request.method == 'POST':
            question = request.get('question')
            if question:
                response = chain(input=question)
                return handle_response(response)
            else:
                return {'error': 'No question provided in the request'}
    except Exception as e:
        return {'error': str(e)}

def handle_response(response):
    if "error" in response:
        return JsonResponse({"error": response["error"]}, status=400)
    else:
        return JsonResponse({"response": response["response"]})