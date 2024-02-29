from dotenv import load_dotenv
from langchain.tools import Tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, messages_to_dict
import json
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories.upstash_redis import UpstashRedisChatMessageHistory
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores.chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
import csv
import logging
import json
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, HttpRequest
from django.views.decorators.http import require_POST
from langchain.agents import AgentExecutor, create_json_chat_agent
load_dotenv()

def read_csv(file_path: str, csv_args: dict = {'delimiter': ','}):
    """Reads a CSV file and returns its contents as a list of strings.

    Args:
        file_path (str): The path to the CSV file.
        csv_args (dict, optional): Additional arguments to be passed to csv.reader. Defaults to {'delimiter': ','}.

    Returns:
        List[str]: A list containing the contents of the CSV file.
    """
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file, **csv_args)
        for row in csv_reader:
            data.append(' '.join(row))
    return data

def chroma_db_tool() -> Chroma:
    logging.info("Initializing Chroma DB tool...")
    try:
        raw_documents = read_csv('assistantweb/data/products.csv')
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        chunks = text_splitter.split_documents(raw_documents)
        embedding_function = OpenAIEmbeddings()
        db = Chroma.from_documents(chunks, embedding_function)
        logging.info(f"Chroma DB initialized successfully with {len(chunks)} chunks.")
        return db
    except Exception as e:
        logging.error(f"Failed to initialize Chroma DB: {e}")
        raise


class SearchTool:
    """Search data in ChromaDB using a given query."""
    
    def __call__(self, query: str) -> str:
        """Searches data in Chroma vector store using a given query.

        Args:
            query (str): The query to search for.

        Returns:
            str: The search result.
        """
        db = chroma_db_tool()
        docs = db.similarity_search(query)
        return docs[0].page_content if docs else "No results found."

search_tool_instance = SearchTool()  

tools = [
    Tool(
        name="chroma_db_search",
        func=search_tool_instance.__call__,
        description="Useful for retrieving data from ChromaDB based on user queries"
    )       
]

model = ChatOpenAI(
    model="gpt-3.5-turbo",
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


prompt = ChatPromptTemplate.from_messages([
    (SystemMessage, {"content": "eres un vendedor de un consecionario de autos nuevos, tu labor es informar y ayudar a los clientes en sus compras con la información a disposición, tienes que responder de forma amable y únicamente con la información que tengas a disposición ya sea de la base de datos o de tu memoria. tienes que dar la respuesta en formato json"}),
    (AIMessage, {"content": "Hola! ¿Cómo puedo ayudarte hoy?"}),
    (HumanMessage, {"content": "{input}"}),
])
json_agent = create_json_chat_agent(
    llm=model,
    prompt=prompt,
    tools=tools
)

agent = AgentExecutor(
    agent=json_agent,
    memory=memory,
    tools=tools,
)

@csrf_exempt
@require_POST
def ask(request: HttpRequest):
    """Handles incoming HTTP POST requests for asking questions.

    Args:
        request (HttpRequest): The HTTP request object containing the question.

    Returns:
        JsonResponse: A JSON response containing the answer to the question.
    """
    try:
        data = json.loads(request.body)
        # question_text = data.get('question') 
        if data:
            # question_content = HumanMessage(content=question_text) 
            response = agent.invoke(data)
            print(response)
            # response_dict =messages_to_dict(response)
            # response.dumps(response)
            # return JsonResponse({"response": response})
        else:
            return JsonResponse({"error": "Question cannot be blank or null"}, status=400)
    except Exception as e:
        print(f"Error occurred: {e}")
        return JsonResponse({'error': str(e)}, status=500)