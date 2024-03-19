# Overview

This project leverages the LangChain framework to create a sophisticated conversational AI agent. It integrates multiple components, such as environment variable management, vector store data retrieval, and a Django-based web interface, to handle user interactions efficiently. The system's modular design facilitates the integration of various conversational AI tools, document loaders, and vector stores, making it highly adaptable and scalable.
Features

## Installation
Prerequisites

    Python 3.8 or higher
    pip (Python package installer)

Setup Instructions

  Create and Activate a Virtual Environment:


Unix/macOS:

    python3 -m venv venv
    source venv/bin/activate

Windows:

    python -m venv venv
    .\venv\Scripts\activate

Install Dependencies:

    pip install -r requirements.txt

Set Environment Variables:

   Create a .env file in the project's root directory.
   Add environment variables such as UPSTASH_URL and UPSTASH_TOKEN with their respective values.

Run the Application:

 Navigate to the project directory.
 Start the Django development server:

        python manage.py runserver

## Usage
Interacting with the Conversational Agent

 The conversational agent can be accessed through a web interface, primarily via a POST request to the /ask endpoint.
 Example using curl:

    curl -X POST \
    http://127.0.0.1:8000/ask \
    -H 'Content-Type: application/json' \
    -d '{"question": "What is the best car?"}'

Understanding the Code Structure

  The project's architecture is modular, with separate components for functionalities such as tool initialization, agent configuration, prompt management, and request handling.
  This design allows for easy customization and extension, enabling developers to tailor conversational flows and data handling mechanisms to meet specific requirements.

Further Reading

  LangChain Documentation: Provides comprehensive information on LangChain tools, agents, and best practices for developing conversational AI applications.
  Django Documentation: Offers in-depth guidance on web application development and deployment using Django, facilitating the integration of web interfaces with conversational agents.
  Vector Space Models: Essential for understanding the principles behind document retrieval and similarity searches, underpinning the project's data retrieval mechanisms.

