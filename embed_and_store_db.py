import dotenv
import os
import requests

from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings

from supabase.client import create_client
from langchain_community.vectorstores import SupabaseVectorStore

dotenv.load_dotenv()
# ——————————————————————————
# 1) HTTP request (nodo HTTP Request)
# ——————————————————————————


def fetch_text(url: str) -> str:
    """Equivalente al nodo HTTP Request de n8n."""
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.text


# ——————————————————————————
# 2) Splitter (Default Data Loader + Recursive Character)
# ——————————————————————————
def split_into_documents(text: str) -> list[Document]:
    """
    Usa RecursiveCharacterTextSplitter para crear fragments.
    create_documents recibe lista de strings y retorna List[Document].
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    return splitter.create_documents([text])


# ——————————————————————————
# 3) Embeddings + Supabase Vector Store
# ——————————————————————————
def embed_and_store(docs: list[Document]):
    # — credenciales Supabase (igual que “Supabase account” en n8n)
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
    supabase = create_client(supabase_url, supabase_key)

    # embeddings OpenAI
    embeddings = OpenAIEmbeddings()

    # vector store configurado igual que en n8n:
    #  • tabla: documentos_rag_2
    #  • batch_size: 200
    vector_store = SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name="documentos_rag_2",
        chunk_size=200,
        query_name="match_documents",
    )
    # inserta los documentos en batch de 200
    vector_store.add_documents(docs)
    print(f"✔️ Almacenados {len(docs)} fragmentos en `documentos_rag_2`.")


if __name__ == "__main__":
    # Trigger manual (equivalente a “When clicking ‘Test workflow’”)
    url = "https://raw.githubusercontent.com/juanhenaoparra/examples/refs/heads/main/truora-blog.txt"
    print("▶️ Obteniendo texto…")
    raw_text = fetch_text(url)

    print("🔪 Dividiendo en documentos…")
    documents = split_into_documents(raw_text)

    print("📥 Generando embeddings y almacenando…")
    embed_and_store(documents)
