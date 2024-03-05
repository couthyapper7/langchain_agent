from dotenv import load_dotenv
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain.prompts import PromptTemplate
import json
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories.upstash_redis import UpstashRedisChatMessageHistory
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores.chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
import csv
import logging
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, HttpRequest
from django.views.decorators.http import require_POST
from langchain.agents import AgentExecutor, create_json_chat_agent
load_dotenv()

def read_csv(file_path: str, csv_args: dict = {'delimiter': ','}):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file, **csv_args)
        for row in csv_reader:
            data.append(' '.join(row))
    return data

def chroma_db_tool() -> Chroma:
    logging.info("Initializing Chroma DB tool...")
    try:
        raw_documents = read_csv('path/to/your/csv')
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        chunks = text_splitter.split_documents(raw_documents)
        embedding_function = OpenAIEmbeddings()
        db = Chroma.from_documents(chunks, embedding_function)
        logging.info(f"Chroma DB initialized successfully with {len(chunks)} chunks.")
        return db
    except Exception as e:
        logging.error(f"Failed to initialize Chroma DB: {e}")
        raise



class SearchTool(BaseTool):
    name = "chroma_db_search"  
    description = "Useful for retrieving data of cars for the user"  
    def _run (self, query: str) -> str:
        """Searches data in Chroma vector store using a given query.

        Args:
            query (str): The query to search for.

        Returns:
            str: The search result.
        """
        db = chroma_db_tool()  
        docs = db.similarity_search(query)
        return docs[0].page_content if docs else "No results found."
    def _arun (self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("this tool does not support async")



tools = [SearchTool()]

model = ChatOpenAI(
    model="gpt-4-1106-preview",
    temperature=0
)

UPSTASH_URL = "https://us1-excited-haddock-41233.upstash.io"
UPSTASH_TOKEN ="AaERACQgMDliYjdlZWYtYzgzYi00NmJiLTk2MjktNTllNjZjZDBjZWYxMmE5OGRiNGRmZTgxNDc2YjkyMmQ1YzExOWE0NTljNjI="
history = UpstashRedisChatMessageHistory(
    url=UPSTASH_URL, token=UPSTASH_TOKEN,ttl=600, session_id="chat1"
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="input",
    return_messages=True
    )

messages = [
    SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template='eres un vendedor de un consecionario que debe ser lo mas amable y amigable posible con el cliente e intentar venderle alguno de nuestros autos a toda costa. tu no tienes ningun conocimiento sobre autos asi que debes responder solamente con los datos proporcionados por tu herramienta: chroma_db_search. puedes decorar esa informacion para hacer un mensaje mas comodo de leer y que parezca una interaccion humana (siempre manteniendo la cordialidad) pero no puedes cambiar los valores recibidos por tu herramienta chroma_db_search')),
    MessagesPlaceholder(variable_name='chat_history', optional=True),
    HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}')),
    MessagesPlaceholder(variable_name='agent_scratchpad'),
    MessagesPlaceholder(variable_name='tools'),
    MessagesPlaceholder(variable_name='tool_names'),
]

prompt = ChatPromptTemplate.from_messages(messages)

json_agent = create_json_chat_agent(
    llm=model,
    prompt=prompt,
    tools=tools
)

agent = AgentExecutor(
    agent=json_agent,
    llm=model,
    max_iterations=3,
    early_stopping_method='generate',
    memory=memory,
    tools=tools,
    input_variables=["input"],
    verbose=True,
    return_intermediate_steps=True,
)


@csrf_exempt
@require_POST
def ask(request: HttpRequest):
    try:
        data = json.loads(request.body)
        question = data.get('question')
        if question:
            response = agent.run({'input': question})
            return JsonResponse({"response": response})
        else:
            return JsonResponse({"error": "Question cannot be blank or null"}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)