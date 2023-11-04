import os
from dotenv import load_dotenv

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader, DirectoryLoader

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

persist_directory = "db"

embedding = OpenAIEmbeddings()

vectordb2 = Chroma(persist_directory=persist_directory, embedding_function=embedding)

retriever = vectordb2.as_retriever(search_kwargs={"k": 2})

# Setup the turbo LLM
turbo_llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

qa_chain = RetrievalQA.from_chain_type(
    llm=turbo_llm, chain_type="stuff", retriever=retriever, return_source_documents=True
)


def process_llm_response(llm_response):
    print(f"Result:\n{llm_response['result']}")
    print("\nSources:")
    for source in llm_response["source_documents"]:
        print(source.metadata["source"])


while True:
    query = input(
        "Enter your question, type 'delete' to delete ChromaDB, or 'exit' to quit: "
    )
    if query.lower() == "exit":
        break
    elif query.lower() == "delete":
        vectordb2.delete_collection()
        vectordb2.persist()
        print("ChromaDB has been deleted.")
    else:
        llm_response = qa_chain(query)
        process_llm_response(llm_response)
