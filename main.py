"""Pre-entrega 3 - Prueba del sistema RAG local: una pregunta con respuesta y una pregunta trampa."""

import asyncio
import logging

from ingest import ingest
from rag import PREGUNTA_EJEMPLO, get_rag_response, retrieve
from schemas import RAGResponse

__all__ = ["RAGResponse", "get_rag_response", "ingest", "main", "retrieve"]

# La respuesta no está en los documentos de data/: el sistema debe decir "No lo sé".
PREGUNTA_TRAMPA = "¿Cuál es el presupuesto anual de la plataforma Atlas?"

PRUEBAS = {
    "Pregunta con respuesta en los documentos": PREGUNTA_EJEMPLO,
    "Pregunta trampa (sin respuesta en los documentos)": PREGUNTA_TRAMPA,
}


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    ingest()

    for nombre, pregunta in PRUEBAS.items():
        print(f"\n=== {nombre} ===\n{pregunta}\n")
        print("Fragmentos recuperados:")
        for documento in await retrieve(pregunta):
            inicio = " ".join(documento.page_content.split())[:70]
            print(f"  - {documento.metadata['source']} (fragmento {documento.metadata['chunk']}): {inicio}...")

        respuesta = await get_rag_response(pregunta)
        print("\nRespuesta:")
        if respuesta is None:
            print("Sin respuesta: el motivo está en el log.")
        else:
            print(respuesta.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
