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
from rest_framework.authtoken.models import Token
import os
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from rest_framework import status, views
from rest_framework.response import Response
from .serializers import UserSerializer
from .jwt_settings import *
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework.permissions import AllowAny
from rest_framework.views import APIView
from .models import *
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
    model='gpt-3.5-turbo-0125',
    temperature=0
)

memory = ConversationBufferMemory(
    memory_key='chat_history',
    input_key='input',
    return_messages=True
)

format_schema = JsonOutputParser()

messages = [
    SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template='eres un vendedor de un consecionario que debe ser lo mas amable y amigable posible con el cliente e intentar venderle alguno de nuestros autos a toda costa. tu no tienes ningun conocimiento sobre autos asi que debes responder solamente con los datos proporcionados por tu herramienta: chroma_db_search. puedes decorar esa informacion para hacer un mensaje mas comodo de leer y que parezca una interaccion humana (siempre manteniendo la cordialidad) pero no puedes cambiar los valores recibidos por tu herramienta chroma_db_search. ten en cuenta que la seccion de submodelo se refiere unicamente al tamaño del motor. y tienes que recordar la conversacion en todo momento si te hacen una pregunta sobre ellos mismos debes responder con los datos que tienes en tu memoria o pedirle que amplien tu informacion para poder conocerse')),
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
        data = json.loads(request.body)
        token = data.get('token')
        if TokenStore.objects.filter(access_token=token).exists():
            question = data.get('question')
            user = data.get(str('user'))
            if user:
                if question:
                    
                    history = UpstashRedisChatMessageHistory(
                    url=UPSTASH_URL, token=UPSTASH_TOKEN,ttl=600, session_id=user
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
                return JsonResponse({'error':'user not provided'})
        else:
            return JsonResponse({'error': 'Authorization token is missing'}, status=403)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)





class UserCreate(views.APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        """
        Creates a new user account and generates a new set of authentication tokens.

        This method handles the POST request to the user creation endpoint. It takes the request data to create a new user through the UserSerializer.
        If the user is successfully created, it generates a refresh token and an access token for the user, stores them, and returns them alongside the user's data.
        If the user creation fails (e.g., due to invalid data), it returns the error details.

        Args:
            request (HttpRequest): The request object containing the data for creating a new user. This includes fields like username, email, and password.

        Returns:
            Response: An HTTP response object. If the user is successfully created, this response includes the HTTP status code 201 (HTTP_201_CREATED), user data, and the authentication tokens. If there's an error (e.g., invalid request data), it returns the errors with the HTTP status code 400 (HTTP_400_BAD_REQUEST).
        """
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            refresh = RefreshToken.for_user(user)  
            access_token = str(refresh.access_token)
            refresh_token = str(refresh)

            TokenStore.objects.create(
                user=user,
                access_token=access_token,  
                refresh_token=refresh_token 
            )
            return Response({
                'user': serializer.data,
                'access_token': access_token,
                'refresh_token': refresh_token
            }, status=status.HTTP_201_CREATED)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)



