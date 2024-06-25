from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
import os

load_dotenv()
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))
SEPARATOR = os.getenv("SEPARATOR")

raw_documents = TextLoader("../data/txt/test.txt", encoding="utf-8").load()

text_splitter = CharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separator=SEPARATOR
)
split_documents = text_splitter.split_documents(raw_documents)
print(split_documents[0].page_content)  #リストの0番目の文字列を表示

# for i, doc in enumerate(split_documents):
#     print("------")
#     print(f"{i}:\n{doc.page_content}")
