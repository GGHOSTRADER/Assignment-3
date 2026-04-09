from langchain_chroma import Chroma
from config import get_embeddings, DB_FOLDER
import os

embeddings = get_embeddings()

# Pick which DB you want
persist_dir = os.path.join(DB_FOLDER, "apple")

vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

query = "Apple Balance Sheet"

docs = retriever.invoke(query)

for i, d in enumerate(docs):
    print(f"\n--- RESULT {i+1} ---")
    print("SOURCE:", d.metadata)
    print("CONTENT:", d.page_content[:500])
