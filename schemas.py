"""Contrato de salida del sistema RAG: el texto de la respuesta y sus referencias."""

from pydantic import BaseModel, Field, field_validator

NO_LO_SE = "No lo sé"


class RAGResponse(BaseModel):
    """Respuesta generada a partir del contexto recuperado."""

    texto: str = Field(
        description=f"Respuesta a la pregunta, o exactamente '{NO_LO_SE}' si no está en el contexto."
    )
    referencias: list[str] = Field(
        default_factory=list,
        description="Nombres de los archivos fuente usados para responder. Lista vacía si no se sabe.",
    )

    @field_validator("texto")
    @classmethod
    def _texto_con_contenido(cls, valor: str) -> str:
        valor = valor.strip()
        if not valor:
            raise ValueError("el texto de la respuesta no puede estar vacío")
        return valor

    @field_validator("referencias")
    @classmethod
    def _referencias_sin_repetir(cls, valor: list[str]) -> list[str]:
        return list(dict.fromkeys(r.strip() for r in valor if r.strip()))

    @property
    def sin_respuesta(self) -> bool:
        return self.texto.lower().startswith(NO_LO_SE.lower())
