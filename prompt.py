from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from redundant_filter_retriever import RedundantFilterRetriever

load_dotenv()

chat = ChatOpenAI()

embeddings = OpenAIEmbeddings()

db = Chroma(
    persist_directory="emb",
    embedding_function=embeddings
)

retriever = RedundantFilterRetriever(
    embeddings=embeddings,
    chroma=db,
)

chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff",
)

result = chain.invoke("what is an interesting fact about the english language?")

print(result)