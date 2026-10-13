"""Cadena RAG asíncrona: recupera fragmentos de ChromaDB y genera una respuesta basada solo en ellos."""

import logging
import os
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough

from ingest import SIN_CLAVE, get_vectorstore, ingest
from schemas import NO_LO_SE, RAGResponse

log = logging.getLogger("rag")

TOP_K = 4  # entre 3 y 5: suficiente contexto sin diluir la atención del modelo
TEMPERATURE = 0
MAX_TOKENS = 1024
MAX_INTENTOS = 2
API_KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}

PREGUNTA_EJEMPLO = "¿Cuántas réplicas mínimas debe tener un servicio crítico en producción?"

parser = PydanticOutputParser(pydantic_object=RAGResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un asistente técnico. Responde solo basándote en el CONTEXTO proporcionado. "
            f"Si la respuesta no está allí, responde exactamente \"{NO_LO_SE}\" y deja las "
            "referencias vacías. No uses conocimiento propio ni completes lo que falte.\n"
            "En referencias, indica los archivos fuente de los fragmentos que usaste.\n\n"
            "{instrucciones_formato}",
        ),
        ("human", "CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}"),
    ]
).partial(instrucciones_formato=parser.get_format_instructions())


def build_model(provider: str | None = None) -> BaseChatModel:
    provider = provider or os.getenv("LLM_PROVIDER") or "openai"
    # Cada integración se importa solo si se usa: alcanza con tener instalada una de las dos.
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL") or "claude-haiku-4-5",
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            api_key=os.getenv("ANTHROPIC_API_KEY") or SIN_CLAVE,
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL") or "gpt-4o-mini",
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        api_key=os.getenv("OPENAI_API_KEY") or SIN_CLAVE,
    )


def format_docs(documentos: list[Document]) -> str:
    """Transforma los fragmentos recuperados en el texto de contexto, cada uno con su fuente."""
    return "\n\n".join(f"[fuente: {d.metadata['source']}]\n{d.page_content}" for d in documentos)


def build_chain(retriever: BaseRetriever, model: BaseChatModel) -> Runnable[str, dict]:
    """Cadena LCEL: retriever -> formato de documentos -> prompt -> modelo -> PydanticOutputParser."""
    generacion = (
        RunnablePassthrough.assign(contexto=lambda x: format_docs(x["documentos"]))
        | prompt
        | model
        | parser
    ).with_retry(retry_if_exception_type=(OutputParserException,), stop_after_attempt=MAX_INTENTOS)
    # Los documentos recuperados viajan junto a la respuesta para poder verificar las referencias.
    return {"documentos": retriever, "pregunta": RunnablePassthrough()} | RunnablePassthrough.assign(
        respuesta=generacion
    )


def build_retriever() -> BaseRetriever:
    vectorstore = get_vectorstore()
    ingest(vectorstore=vectorstore)  # no reindexa si la colección ya está poblada
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})


@lru_cache(maxsize=1)
def get_chain() -> Runnable[str, dict]:
    return build_chain(build_retriever(), build_model())


async def retrieve(query: str = PREGUNTA_EJEMPLO) -> list[Document]:
    """Capa de recuperación: convierte la pregunta en un embedding y trae los fragmentos más cercanos."""
    return await build_retriever().ainvoke(query)


def _error_de_configuracion() -> str | None:
    provider = os.getenv("LLM_PROVIDER") or "openai"
    variable = API_KEY_ENV.get(provider)
    if variable is None:
        return f"LLM_PROVIDER debe ser 'openai' o 'anthropic', no '{provider}'"
    if not os.getenv(variable):
        return f"Falta la variable de entorno {variable}"
    return None


async def get_rag_response(query: str = PREGUNTA_EJEMPLO) -> RAGResponse | None:
    """Responde una consulta con el flujo RAG completo. Devuelve None, con el motivo en el log, si falla."""
    if not query or not query.strip():
        log.error("La consulta está vacía")
        return None
    if error := _error_de_configuracion():
        log.error(error)
        return None

    log.info("Consulta: %s", query)
    try:
        salida = await get_chain().ainvoke(query)
    except OutputParserException as exc:
        log.error("El modelo no devolvió el formato pedido tras %d intentos: %s", MAX_INTENTOS, str(exc).splitlines()[0])
        return None
    except Exception as exc:
        # Clave inválida, límite de tasa o red, ya agotados los reintentos del SDK.
        log.error("Falló la llamada al modelo (%s): %s", type(exc).__name__, exc)
        return None

    respuesta: RAGResponse = salida["respuesta"]
    recuperadas = list(dict.fromkeys(d.metadata["source"] for d in salida["documentos"]))
    log.info("Fragmentos recuperados: %d, de %s", len(salida["documentos"]), recuperadas)

    # Solo se aceptan referencias a fuentes que realmente estuvieron en el contexto.
    validas = [r for r in respuesta.referencias if r in recuperadas]
    if len(validas) != len(respuesta.referencias):
        log.warning("Se descartaron referencias que no estaban en el contexto: %s",
                    [r for r in respuesta.referencias if r not in recuperadas])
    if respuesta.sin_respuesta:
        validas = []
        log.info("El contexto no contiene la respuesta")
    return respuesta.model_copy(update={"referencias": validas})
