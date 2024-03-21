from dotenv import load_dotenv
from django.contrib.auth import authenticate
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
import json
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories.upstash_redis import UpstashRedisChatMessageHistory
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.views.decorators.http import require_POST,require_GET
from langchain.agents import AgentExecutor, create_json_chat_agent
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun,CallbackManagerForToolRun
import logging
from langchain.tools.render import render_text_description
from typing import Optional
import os
from .jwt_settings import *
load_dotenv()


UPSTASH_URL = os.getenv("UPSTASH_URL")
UPSTASH_TOKEN = os.getenv("UPSTASH_TOKEN")


def chroma_db_tool() -> Chroma:
    """Initialize the Chroma DB tool.

    Returns:
        Chroma: The Chroma database object.
    """
    logging.info("Initializing Chroma DB tool...")
    try:
        embedding_function = OpenAIEmbeddings()
        loader = CSVLoader("/home/fausto/assistantweb/data/products.csv")  
        documents = loader.load()
        db = Chroma.from_documents(documents, embedding_function)
        
        logging.info(f"Chroma DB initialized successfully with {len(documents)} documents.")
        return db
    except Exception as e:
        logging.error(f"Failed to initialize Chroma DB: {e}")
        raise

global chroma_db
chroma_db = chroma_db_tool()

class SearchTool(BaseTool):
    name = "chroma_db_search"  
    description = "Useful for retrieving data of cars for the user"  
    def _run (self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Searches data in Chroma vector store using a given query.

        Args:
            query (str): The query to search for.

        Returns:
            str: The search result.
        """
        db = chroma_db
        docs = db.similarity_search(query)
        return docs[0].page_content if docs else "No results found."
    def _arun (self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("this tool does not support async")

tools = [SearchTool()]



model = ChatOpenAI(
    model='GPT-3.5 Turbo',
    temperature=0
)

memory = ConversationBufferMemory(
    memory_key='chat_history',
    input_key='input',
    return_messages=True
)

format_schema = JsonOutputParser()

messages = [
    SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template='eres un vendedor de un consecionario que debe ser lo mas amable y amigable posible con el cliente e intentar venderle alguno de nuestros autos a toda costa. tu no tienes ningun conocimiento sobre autos asi que debes responder solamente con los datos proporcionados por tu herramienta: chroma_db_search. puedes decorar esa informacion para hacer un mensaje mas comodo de leer y que parezca una interaccion humana (siempre manteniendo la cordialidad) pero no puedes cambiar los valores recibidos por tu herramienta chroma_db_search. ten en cuenta que la seccion de submodelo se refiere unicamente al tamaño del motor')),
    MessagesPlaceholder(variable_name='chat_history', optional=True),
    HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template=
    """TOOLS
    ------
    Assistant can ask the user to use tools to look up information that may be helpful in answering the users original question. The tools the human can use are:
    {tools}
    RESPONSE FORMAT INSTRUCTIONS
    ----------------------------
    When responding to me, please output a response in one of two formats:
    **Option 1:**
    Use this if you want the human to use a tool.
    Markdown code snippet formatted in the following schema:
    
    {{
        "action": string, \ The action to take. Must be one of {tool_names}
        "action_input": string \ The input to the action
    }}
    Option 2:
    Use this if you want to respond directly to the human. Markdown code snippet formatted in the following schema:

    {{

        "action": "Final Answer",

        "action_input": string \ You should put what you want to return to use here

    }}
    USER'S INPUT

    --------------------

    Here is the user's input (remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else):

    {input}
    """
        )),
    MessagesPlaceholder(variable_name='agent_scratchpad')
]

prompt = ChatPromptTemplate.from_messages(messages)

tools_description = render_text_description(list(tools))
tool_names = ', '.join([t.name for t in tools])


json_agent = create_json_chat_agent(
    llm=model,
    prompt=prompt,
    tools=tools
)

@csrf_exempt
@require_POST
def ask(request):
    """manage the method to prompt to the agent

    Args:
        request (str): (from question) prompt

    Returns:
        json: agent response
    """
    try:
        token = request.headers.get('Authorization')
        if token:
            data = json.loads(request.body)
            question = data.get('question')
            if question:
                
                history = UpstashRedisChatMessageHistory(
                url=UPSTASH_URL, token=UPSTASH_TOKEN,ttl=600, session_id=token
                )
                
                agent = AgentExecutor(
                agent=json_agent,
                llm=model,
                max_iterations=3,
                early_stopping_method='generate',
                memory=history,
                tools=tools,
                input_variables=['input'],
                verbose=True,
                return_intermediate_steps=False,
                )
                
                inputs = {'input': question}
                result = agent.invoke(inputs)  
                if 'output' in result:
                    response_content = result['output']
                    return JsonResponse({'response': response_content})
                else:
                    return JsonResponse({'error': 'Expected output key not found', 'raw_output': str(result)}, status=500)
            else:
                return JsonResponse({'error': 'Question cannot be blank or null'}, status=400)
        else:
            return JsonResponse({'error': 'Authorization token is missing'}, status=403)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_POST
def refresh_token(request):
    data = json.loads(request.body)
    refresh_token = data.get('refresh_token')
    if not refresh_token:
        return JsonResponse({'error': 'Refresh token is missing'}, status=400)
    new_access_token = refresh_access_token(refresh_token)
    if isinstance(new_access_token, str) and not new_access_token.startswith('ey'):
        return JsonResponse({'error': new_access_token}, status=403)

    return JsonResponse({'access_token': new_access_token})



@csrf_exempt
@require_POST
def create_token(request):
    try:
        data = json.loads(request.body)
        username = data.get('username')
        password = data.get('password')

        user = authenticate(username=username, password=password)
        print(user)
        print(username)
        if user:
            access_token = generate_token(user.id)
            refresh_token = generate_refresh_token(user.id)
            return JsonResponse({
                'access_token': access_token,
                'refresh_token': refresh_token,
            })
        else:
            return JsonResponse({'error': 'Invalid credentials'}, status=401)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)