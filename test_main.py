"""Tests de la pre-entrega 3. No usan la red ni claves: los embeddings y el modelo son dobles.

Ejecutar con: pytest -v
"""

import json
import logging
from typing import Any

import pytest
import tiktoken
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ValidationError

import ingest as ingesta
import main
import rag
from schemas import RAGResponse

pytestmark = pytest.mark.asyncio

PREGUNTA = "¿Cuántas réplicas mínimas debe tener un servicio crítico en producción?"


def salida(texto: str, referencias: list[str]) -> str:
    return json.dumps({"texto": texto, "referencias": referencias}, ensure_ascii=False)


class ModeloFalso(BaseChatModel):
    """Devuelve las respuestas preparadas, en orden, y guarda los mensajes que recibió."""

    respuestas: list[Any]
    llamadas: int = 0
    recibido: list = []

    @property
    def _llm_type(self) -> str:
        return "falso"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.llamadas += 1
        self.recibido = messages
        siguiente = self.respuestas[min(self.llamadas, len(self.respuestas)) - 1]
        if isinstance(siguiente, Exception):
            raise siguiente
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=siguiente))])


@pytest.fixture
def vectorstore(tmp_path):
    """Colección vacía en una carpeta temporal, con embeddings falsos y deterministas."""
    return ingesta.get_vectorstore(DeterministicFakeEmbedding(size=32), persist_dir=tmp_path / "vs")


@pytest.fixture
def con_clave(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "clave-de-prueba")


@pytest.fixture
def sistema(monkeypatch, vectorstore, con_clave):
    """Arma el flujo completo sobre la colección temporal; devuelve una función que fija el modelo."""
    ingesta.ingest(vectorstore=vectorstore)
    retriever = vectorstore.as_retriever(search_kwargs={"k": rag.TOP_K})

    def con_modelo(*respuestas) -> ModeloFalso:
        modelo = ModeloFalso(respuestas=list(respuestas))
        cadena = rag.build_chain(retriever, modelo)
        monkeypatch.setattr(rag, "get_chain", lambda: cadena)
        return modelo

    return con_modelo


# --- Estrategia de chunking e ingesta ---


async def test_carga_los_documentos_de_data_con_su_fuente():
    documentos = ingesta.cargar_documentos()

    assert [d.metadata["source"] for d in documentos] == ["arquitectura.md", "despliegues.md", "incidentes.md"]
    assert all(d.page_content for d in documentos)


async def test_limpiar_quita_espacios_finales_y_saltos_repetidos():
    assert ingesta.limpiar("uno  \n\n\n\ndos\t\n") == "uno\n\ndos"


async def test_los_fragmentos_respetan_el_tamano_en_tokens_y_se_solapan():
    fragmentos = ingesta.fragmentar(ingesta.cargar_documentos())
    tokens = tiktoken.get_encoding("cl100k_base")
    de_despliegues = [f.page_content for f in fragmentos if f.metadata["source"] == "despliegues.md"]

    assert all(len(tokens.encode(f.page_content)) <= ingesta.CHUNK_SIZE for f in fragmentos)
    assert len(de_despliegues) >= 2  # el documento no entra en un solo fragmento
    # Overlap: el inicio del segundo fragmento ya estaba al final del primero.
    assert de_despliegues[1].splitlines()[0] in de_despliegues[0]


async def test_los_ids_son_deterministas_y_unicos():
    primera = [f.id for f in ingesta.fragmentar(ingesta.cargar_documentos())]
    segunda = [f.id for f in ingesta.fragmentar(ingesta.cargar_documentos())]

    assert primera == segunda
    assert len(set(primera)) == len(primera)
    assert primera[0].startswith("arquitectura.md:0:")


async def test_ingest_persiste_en_disco_y_no_reindexa_si_ya_existe(vectorstore, tmp_path, caplog):
    cantidad = ingesta.ingest(vectorstore=vectorstore)

    with caplog.at_level(logging.INFO, logger="rag.ingest"):
        repetida = ingesta.ingest(vectorstore=vectorstore)
    reabierta = ingesta.get_vectorstore(DeterministicFakeEmbedding(size=32), persist_dir=tmp_path / "vs")

    assert cantidad == repetida > 0
    assert "no se vuelve a indexar" in caplog.text
    assert len(reabierta.get(include=[])["ids"]) == cantidad  # sobrevivió en disco


async def test_ingest_con_force_reindexa_sin_duplicar(vectorstore):
    cantidad = ingesta.ingest(vectorstore=vectorstore)

    assert ingesta.ingest(force=True, vectorstore=vectorstore) == cantidad
    assert len(vectorstore.get(include=[])["ids"]) == cantidad


async def test_ingest_sin_archivos_no_falla(vectorstore, tmp_path):
    assert ingesta.ingest(data_dir=tmp_path, vectorstore=vectorstore) == 0


async def test_cada_modelo_de_embeddings_tiene_su_propia_coleccion(monkeypatch, tmp_path):
    falso = DeterministicFakeEmbedding(size=32)
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "local")
    local = ingesta.get_vectorstore(falso, persist_dir=tmp_path)._collection.name
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "openai")
    openai = ingesta.get_vectorstore(falso, persist_dir=tmp_path)._collection.name

    assert local == "atlas_all_minilm_l6_v2"
    assert openai == "atlas_text_embedding_3_small"


async def test_sin_clave_de_openai_los_embeddings_son_locales(monkeypatch):
    monkeypatch.delenv("EMBEDDINGS_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert ingesta.embeddings_provider() == "local"

    monkeypatch.setenv("OPENAI_API_KEY", "clave-de-prueba")
    assert ingesta.embeddings_provider() == "openai"


# --- Recuperación y arquitectura asíncrona ---


async def test_el_retriever_devuelve_top_k_fragmentos(vectorstore):
    ingesta.ingest(vectorstore=vectorstore)

    documentos = await vectorstore.as_retriever(search_kwargs={"k": rag.TOP_K}).ainvoke(PREGUNTA)

    assert len(documentos) == rag.TOP_K
    assert 3 <= rag.TOP_K <= 5
    assert all("source" in d.metadata for d in documentos)


async def test_el_prompt_exige_responder_solo_con_el_contexto():
    mensajes = rag.prompt.format_messages(contexto="CTX", pregunta="PREG")

    assert sorted(rag.prompt.input_variables) == ["contexto", "pregunta"]
    assert "solo basándote en el CONTEXTO" in mensajes[0].content
    assert '"No lo sé"' in mensajes[0].content
    assert rag.parser.get_format_instructions() in mensajes[0].content
    assert "CTX" in mensajes[1].content and "PREG" in mensajes[1].content


async def test_format_docs_incluye_la_fuente_de_cada_fragmento():
    texto = rag.format_docs([Document(page_content="contenido", metadata={"source": "a.md"})])

    assert texto == "[fuente: a.md]\ncontenido"


async def test_get_rag_response_devuelve_texto_y_referencias(sistema):
    modelo = sistema(salida("Un mínimo de 3 réplicas.", ["despliegues.md"]))

    respuesta = await rag.get_rag_response(PREGUNTA)

    assert isinstance(respuesta, RAGResponse)
    assert respuesta.texto == "Un mínimo de 3 réplicas."
    assert respuesta.referencias == ["despliegues.md"]
    # El modelo recibió los fragmentos recuperados y la pregunta dentro del prompt.
    assert "[fuente: " in modelo.recibido[1].content
    assert PREGUNTA in modelo.recibido[1].content


# --- Validación con Pydantic ---


async def test_rag_response_valida_texto_y_referencias():
    respuesta = RAGResponse(texto="  Tres réplicas. ", referencias=["a.md", "a.md", " "])

    assert respuesta.texto == "Tres réplicas."
    assert respuesta.referencias == ["a.md"]
    with pytest.raises(ValidationError):
        RAGResponse(texto="   ")


async def test_reintenta_si_la_salida_no_cumple_el_esquema(sistema):
    modelo = sistema("esto no es JSON", salida("Un mínimo de 3 réplicas.", ["despliegues.md"]))

    respuesta = await rag.get_rag_response(PREGUNTA)

    assert modelo.llamadas == 2
    assert respuesta.texto == "Un mínimo de 3 réplicas."


async def test_devuelve_none_si_el_modelo_nunca_cumple_el_formato(sistema, caplog):
    modelo = sistema("esto no es JSON")

    with caplog.at_level(logging.ERROR, logger="rag"):
        respuesta = await rag.get_rag_response(PREGUNTA)

    assert respuesta is None
    assert modelo.llamadas == rag.MAX_INTENTOS
    assert "no devolvió el formato pedido" in caplog.text


# --- Robustez y evidencia de pruebas ---


async def test_pregunta_trampa_responde_no_lo_se_sin_referencias(sistema):
    sistema(salida("No lo sé", ["despliegues.md"]))

    respuesta = await rag.get_rag_response("¿Cuál es el presupuesto anual de la plataforma Atlas?")

    assert respuesta.texto == "No lo sé"
    assert respuesta.sin_respuesta
    assert respuesta.referencias == []


async def test_descarta_referencias_que_no_estaban_en_el_contexto(sistema, caplog):
    sistema(salida("Un mínimo de 3 réplicas.", ["despliegues.md", "inventado.md"]))

    with caplog.at_level(logging.WARNING, logger="rag"):
        respuesta = await rag.get_rag_response(PREGUNTA)

    assert respuesta.referencias == ["despliegues.md"]
    assert "inventado.md" in caplog.text


async def test_devuelve_none_si_falla_la_api(sistema):
    sistema(ConnectionError("sin red"))

    assert await rag.get_rag_response(PREGUNTA) is None


@pytest.mark.parametrize("consulta", ["", "   "])
async def test_rechaza_una_consulta_vacia(con_clave, consulta):
    assert await rag.get_rag_response(consulta) is None


async def test_sin_argumentos_ni_clave_no_se_rompe(monkeypatch, caplog):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with caplog.at_level(logging.ERROR, logger="rag"):
        respuesta = await rag.get_rag_response()

    assert respuesta is None
    assert "Falta la variable de entorno OPENAI_API_KEY" in caplog.text


async def test_las_claves_no_estan_en_el_repositorio():
    plantilla = (ingesta.BASE / ".env.example").read_text()
    ignorados = (ingesta.BASE / ".gitignore").read_text().split()

    assert "OPENAI_API_KEY=\n" in plantilla and "ANTHROPIC_API_KEY=\n" in plantilla
    assert ".env" in ignorados and "vectorstore/" in ignorados


async def test_main_corre_las_dos_pruebas_sin_clave(monkeypatch, capsys):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    async def recuperar(pregunta):
        return [Document(page_content="texto de prueba", metadata={"source": "a.md", "chunk": 0})]

    monkeypatch.setattr(main, "ingest", lambda: 0)
    monkeypatch.setattr(main, "retrieve", recuperar)

    await main.main()

    impreso = capsys.readouterr().out
    assert "Pregunta trampa" in impreso and "a.md (fragmento 0)" in impreso
    assert impreso.count("Sin respuesta: el motivo está en el log.") == 2
