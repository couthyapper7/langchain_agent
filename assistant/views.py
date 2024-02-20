from dotenv import load_dotenv
from langchain.tools import Tool
from .tools import CSVSearchTool,csv_path
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_community.chat_message_histories.upstash_redis import UpstashRedisChatMessageHistory
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
import chromadb
load_dotenv()

chroma_client = chromadb.Client()


tools = [
    Tool(
        name="chroma_db_search",
        func=chroma_db_tool.run,
        description="Useful for retrieving data from ChromaDB based on user queries."
    )
]


tools = [
    Tool(
        name="csv_search",
        func=CSVSearchTool.run,
        description="Useful for when you need to answer questions about cars based on a CSV database."
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
    tools=tools
)


def ask(request):
    try:
        if request.method == 'POST':
            question = request.json.get('question')
            if question:
                response = chain(input=question)
                return {'response': response}
            else:
                return {'error': 'No question provided in the request'}
    except Exception as e:
        return {'error': str(e)}