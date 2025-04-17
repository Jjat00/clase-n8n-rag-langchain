import dotenv
import os
import logging

# —— Telegram ——
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    ContextTypes,
    filters,
)

# —— LangChain ——
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# —— Supabase ——
from supabase.client import create_client
from langchain_community.vectorstores import SupabaseVectorStore

dotenv.load_dotenv()

# —————————————
# Configuración
# —————————————
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# —————————————
# 1) Inicializar Vector Store + Retriever
# —————————————
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
embeddings = OpenAIEmbeddings()
vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documentos_rag_2",
    chunk_size=200,
    query_name="match_documents_lc",
)

retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# —————————————
# 2) QA Chain
# —————————————
llm = ChatOpenAI(temperature=0, model="gpt-4o")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False,
)

# —————————————
# 3) Handler de Telegram
# —————————————


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    if text.startswith("/"):
        # filtramos comandos
        return

    logger.info("🔍 Query: %s", text)

    # docs = retriever.get_relevant_documents(text)
    # print("docs", docs)
    # if not docs:
    #     answer = "Disculpa, no tengo información relevante para esa consulta."
    try:
        response = await qa_chain.ainvoke({"query": text}, return_only_outputs=True)
        answer = response["result"]
    except Exception as e:
        logger.error("❌ Error en QA: %s", e)
        answer = "Lo siento, ocurrió un error."

    await update.message.reply_text(answer)

# —————————————
# 4) Arrancar el bot con ApplicationBuilder
# —————————————


def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Mensajes de texto que no sean comandos
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    logger.info("🤖 Bot arrancado, esperando mensajes…")
    app.run_polling()


if __name__ == "__main__":
    main()
