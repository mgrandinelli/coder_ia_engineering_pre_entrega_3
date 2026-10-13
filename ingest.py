"""Ingesta: lee los documentos de data/, los fragmenta y los guarda en ChromaDB local.

Uso: python ingest.py [--force]   (--force borra la colección y vuelve a indexar)
"""

import hashlib
import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

log = logging.getLogger("rag.ingest")

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
PERSIST_DIR = BASE / "vectorstore"
CHUNK_SIZE = 500  # tokens
CHUNK_OVERLAP = 50  # tokens
SIN_CLAVE = "sin-configurar"  # permite construir los clientes e importar el módulo sin credenciales


class LocalEmbeddings(Embeddings):
    """Embeddings locales con el modelo que trae ChromaDB (all-MiniLM-L6-v2). No usan API key."""

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._funcion = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in vector] for vector in self._funcion(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def embeddings_provider() -> str:
    """'openai' o 'local'. Sin EMBEDDINGS_PROVIDER, usa OpenAI solo si hay clave configurada."""
    elegido = os.getenv("EMBEDDINGS_PROVIDER")
    if elegido in ("openai", "local"):
        return elegido
    return "openai" if os.getenv("OPENAI_API_KEY") else "local"


def embeddings_model() -> str:
    if embeddings_provider() == "openai":
        return os.getenv("OPENAI_EMBEDDINGS_MODEL") or "text-embedding-3-small"
    return "all-MiniLM-L6-v2"


def build_embeddings() -> Embeddings:
    """Único punto donde se crea el modelo de embeddings: lo usan la ingesta y la consulta."""
    if embeddings_provider() == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=embeddings_model(), api_key=os.getenv("OPENAI_API_KEY") or SIN_CLAVE
        )
    return LocalEmbeddings()


def get_vectorstore(embeddings: Embeddings | None = None, persist_dir: Path = PERSIST_DIR) -> Chroma:
    # Una colección por modelo de embeddings: los vectores de modelos distintos no se mezclan.
    coleccion = "atlas_" + re.sub(r"[^a-z0-9]+", "_", embeddings_model().lower()).strip("_")
    return Chroma(
        collection_name=coleccion,
        embedding_function=embeddings or build_embeddings(),
        persist_directory=str(persist_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def limpiar(texto: str) -> str:
    """Quita el ruido que empeora los embeddings: espacios al final y saltos de línea repetidos."""
    texto = re.sub(r"[ \t]+\n", "\n", texto)
    return re.sub(r"\n{3,}", "\n\n", texto).strip()


def cargar_documentos(data_dir: Path = DATA_DIR) -> list[Document]:
    archivos = sorted(p for p in Path(data_dir).glob("*") if p.suffix in (".md", ".txt"))
    return [
        Document(page_content=limpiar(p.read_text(encoding="utf-8")), metadata={"source": p.name})
        for p in archivos
    ]


def build_splitter() -> RecursiveCharacterTextSplitter:
    try:
        # El tamaño se mide en tokens reales, que es como cobran y limitan los modelos.
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base", chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
    except Exception as exc:
        # Sin el tokenizador (por ejemplo, sin red para descargarlo) se aproxima a 4 caracteres por token.
        log.warning("tiktoken no disponible (%s); se fragmenta por caracteres", type(exc).__name__)
        return RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE * 4, chunk_overlap=CHUNK_OVERLAP * 4)


def fragmentar(documentos: list[Document]) -> list[Document]:
    """Recursive Character Splitting: corta por párrafos, luego líneas, luego espacios."""
    fragmentos = build_splitter().split_documents(documentos)
    por_fuente: dict[str, int] = {}
    for fragmento in fragmentos:
        fuente = fragmento.metadata["source"]
        indice = por_fuente.get(fuente, 0)
        por_fuente[fuente] = indice + 1
        huella = hashlib.sha1(fragmento.page_content.encode("utf-8")).hexdigest()[:10]
        fragmento.metadata["chunk"] = indice
        # Id determinista: reindexar el mismo contenido actualiza el registro, no lo duplica.
        fragmento.id = f"{fuente}:{indice}:{huella}"
    return fragmentos


def ingest(data_dir: Path = DATA_DIR, force: bool = False, vectorstore: Chroma | None = None) -> int:
    """Puebla la colección y devuelve cuántos fragmentos tiene. Si ya está poblada, no reindexa."""
    vectorstore = vectorstore or get_vectorstore()
    if force:
        vectorstore.reset_collection()
    elif existentes := len(vectorstore.get(include=[])["ids"]):
        log.info("La colección ya tiene %d fragmentos; no se vuelve a indexar", existentes)
        return existentes

    documentos = cargar_documentos(data_dir)
    if not documentos:
        log.error("No hay archivos .md o .txt en %s", data_dir)
        return 0
    fragmentos = fragmentar(documentos)
    vectorstore.add_documents(fragmentos, ids=[f.id for f in fragmentos])
    log.info(
        "Ingesta lista: %d documentos, %d fragmentos, embeddings %s",
        len(documentos), len(fragmentos), embeddings_model(),
    )
    return len(fragmentos)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    ingest(force="--force" in sys.argv)
