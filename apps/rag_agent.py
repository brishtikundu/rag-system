from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader,PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st


### data in st session 

if "documents_uploaded" not in st.session_state:
    st.session_state.documents_uploaded=False
if "agents" not in st.session_state:
    st.session_state.agents=None
if "vector_store" not in st.session_state:
    st.session_state.vector_store=None
if "messages" not in st.session_state:
    st.session_state.messages=[]
    

def process_documents(path):

    # Load the documents
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.split_documents(docs)

    # Embedding and Vector DB
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview"
    )

    vector_db = InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=embeddings
    )

    # Create Agents - tools, LLM, prompts
    llm = ChatGroq(
        model="openai/gpt-oss-20b"
    )

    @tool
    def retrieve_docs(query: str):
        """
        Retrieve relevant documents to a query from the knowledge base.
        """
        context = ""

        docs = vector_db.similarity_search(
            query=query,
            k=3
        )

        for doc in docs:
            context += doc.page_content + "\n\n"

        return context

    system_prompt = """
    You are a helpful assistant that answers questions using retrieved context.
    My knowledge base consists of the details from the uploaded document.
    ALWAYS use the `retrieve_docs` tool for questions requiring external knowledge.
    """

    memory = InMemorySaver()

    agents = create_agent(
        model=llm,
        tools=[retrieve_docs],
        system_prompt=system_prompt,
        checkpointer=memory
    )

    st.session_state.agents=agents
    st.session_state.documents_uploaded=True


## while True:
##     query=input("user ")
##     if query.lower() in ["exit","quit"]:
##         break
##     response=agents.invoke(
##         {"messages":[{"role":"user","content":query}]},
##         {"configurable":{"thread_id":1}}
##     )
##     result=response["messages"][-1].content
## 
##     print("AI:", result)
    



## upload Ui
if not st.session_state.documents_uploaded:
    uploaded=st.file_uploader(label="select pdf files",type=["pdf"],accept_multiple_files=True)
    if uploaded:
        with st.spinner("Processing..."):
            path="./docs_files/"
            for file in uploaded:
                with open(path+file.name,"wb") as f:
                    f.write(file.getvalue())
            process_documents(path)
            st.rerun()

## Chat Ui

if  st.session_state.documents_uploaded and st.session_state.agents:

    for message in st.session_state.messages:
        role=message.get("role")
        content=message.get("content")
        st.chat_message(role).markdown(content)
    
    query=st.chat_input("Ask anything related to the uploaded documents...")
    if query:
        st.session_state.messages.append({"role":"user","content":query})

        st.chat_message(query)
        response=st.session_state.agents.invoke(
            {"messages":[{"role":"user","content":query}]},
            {"configurable":{"thread_id":1}}
        )
        answer=response["messages"][-1].content
        st.chat_message("ai").markdown(answer)
        st.session_state.messages.append({"role":"ai","content":answer})

    