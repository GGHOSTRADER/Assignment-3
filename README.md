Tree
.
├── README.md
├── __pycache__
├── build_rag.py
├── build_rag_simple.py
├── chroma_db
├── chunk_testing.py
├── config.py
├── control_flow_assig3.mmd
├── data
├── evaluator.py
├── langgraph_agent.py
├── llm_utils.py
└── requirements.txt

FILES
-------------------
build_rag.py

Main builder of Rag, is table aware and can rewrite files. It needs flag --udpdate_doc to run. Has chunk size and overlap.

Example of CLI
python build_rag.py --update_doc yes

-------------------
build_rag_simple.py

Simple text cleaner, to chunks to embeddings and DB. Only relies on chainging the chunk size and overlap.

-------------------

chunk_testing.py

Used to search in the RAG DB and test how different sized and models would affect. Can change the DB it looks for and question it uses for query.

-------------------

config.py

Model groq was added but it broke langChain dependencies pn the environment if I tried to use it directly, so I had to use openai version and point it to Groq url. Must change the embedding model in it to use different models.

-------------------

.env
New provider was added, must have this to run

LLM_PROVIDER= groq 
GROQ_API_KEY= xxxxxxxxxxxxxxxxxx
GROQ_MODEL= meta-llama/llama-4-scout-17b-16e-instruct


-------------------

requirements.txtz

Updated list of requirements I used.


-------------------

llm_utils.py

When trying to use gemini, I added a wrapper to delay the calls and not hit minute limit but overcomed the issue when changed to Groq.