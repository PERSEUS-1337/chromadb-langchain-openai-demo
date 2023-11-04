import os
from dotenv import load_dotenv

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader, DirectoryLoader

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load persisted database from disk
persist_directory = "db"
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# Create the retriever from db
# retriever = vectordb.as_retriever()
# docs = retriever.get_relevant_documents(
#     "How would AI be useful as Virtual Health Assistants?"
# )

# print(len(docs))
# for i, text in enumerate(docs):
#     print(f"{i}: {text}")

retriever = vectordb.as_retriever(search_kwards={"k": 2})

# Creating the Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True
)


# Citing the sources
def process_llm_response(llm_response):
    print(f"Result:\n{llm_response['result']}")
    print("\nSources:")
    for source in llm_response["source_documents"]:
        print(source.metadata["source"])


# Full Example
# query = "What are the main goals of UP as an academic institution?"
# llm_response = qa_chain(query)
# process_llm_response(llm_response)

while True:
    query = input("Enter your question or type 'exit' to quit: ")
    if query.lower() == "exit":
        break

    llm_response = qa_chain(query)
    process_llm_response(llm_response)
