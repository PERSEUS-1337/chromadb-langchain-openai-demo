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

# Load and Process text files
loader = DirectoryLoader("./text", glob="./*.txt", loader_cls=TextLoader)

documents = loader.load()
print(documents)

# Split texts into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
text_chunks = text_splitter.split_documents(documents)

print(len(text_chunks))
for i, chunk in enumerate(text_chunks):
    print(f"{i}: {chunk}\n")

# We then create the DB
# Supply the directory, which is /db, where we will embed and store the texts
persist_directory = "db"

embedding = OpenAIEmbeddings()

vectordb = Chroma.from_documents(
    documents=text_chunks, embedding=embedding, persist_directory=persist_directory
)

vectordb.persist()
vectordb = None
