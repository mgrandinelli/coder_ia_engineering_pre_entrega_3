# Pre-entrega 3: Sistema de recuperación semántica local (RAG)

Curso AI Engineering (Coderhouse), módulo 3.

Un flujo RAG de punta a punta. Recibe una consulta, busca los fragmentos relevantes en una base vectorial local (ChromaDB) y genera una respuesta que usa exclusivamente esa información. Si la respuesta no está en los documentos, el sistema responde "No lo sé".

```python
respuesta = await get_rag_response("¿Cuántas réplicas mínimas debe tener un servicio crítico en producción?")
# RAGResponse(texto="...", referencias=["despliegues.md"])
```

## Cómo ejecutarlo

Requiere Python 3.12 o superior.

```bash
git clone https://github.com/mgrandinelli/coder_ia_engineering_pre_entrega_3.git
cd coder_ia_engineering_pre_entrega_3

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # y completar la clave del proveedor elegido

python ingest.py          # fragmenta los documentos de data/ y los guarda en ./vectorstore
python main.py            # hace las dos pruebas: una pregunta con respuesta y una pregunta trampa
```

`python main.py` también ejecuta la ingesta si la base todavía no existe, así que el segundo comando alcanza.

## Variables de entorno

| Variable | Obligatoria | Qué es |
|---|---|---|
| `LLM_PROVIDER` | No (por defecto `openai`) | Proveedor del modelo de lenguaje: `openai` o `anthropic` |
| `OPENAI_API_KEY` | Con OpenAI | Clave de la API de OpenAI |
| `ANTHROPIC_API_KEY` | Con Anthropic | Clave de la API de Anthropic |
| `OPENAI_MODEL` | No (por defecto `gpt-4o-mini`) | Modelo de OpenAI |
| `ANTHROPIC_MODEL` | No (por defecto `claude-haiku-4-5`) | Modelo de Anthropic |
| `EMBEDDINGS_PROVIDER` | No | `openai` o `local`. Vacío: `openai` si hay `OPENAI_API_KEY`, `local` en caso contrario |

Las claves se leen del archivo `.env`, que está en `.gitignore` junto con la carpeta `vectorstore/`. Al repositorio solo se sube `.env.example`, sin valores.

## Archivos

| Archivo | Contenido |
|---|---|
| `data/` | Dataset de ejemplo: tres documentos `.md` sobre una plataforma de pagos ficticia (arquitectura, despliegues e incidentes) |
| `ingest.py` | Script de ingesta: carga, limpieza, chunking y persistencia en ChromaDB |
| `rag.py` | Cadena RAG asíncrona y la función `get_rag_response(query)` |
| `schemas.py` | Modelo Pydantic de salida: `RAGResponse` (`texto` y `referencias`) |
| `main.py` | Script de prueba con las dos preguntas |
| `test_main.py` | 24 tests, sin red ni claves |

## Arquitectura y decisiones de diseño

El sistema tiene dos flujos separados:

- **Indexación (una vez):** archivos de `data/` → limpieza → chunking → embeddings → colección de ChromaDB en `./vectorstore`
- **Consulta (en cada pregunta):** pregunta → embedding con el mismo modelo → búsqueda de similitud en ChromaDB → los 4 fragmentos más cercanos → prompt con contexto → modelo de lenguaje → `PydanticOutputParser` → `RAGResponse`

### Módulo de ingesta (`ingest.py`)

1. `cargar_documentos()` lee los `.md` y `.txt` de `data/` y guarda el nombre del archivo en el metadato `source`.
2. `limpiar()` quita los espacios al final de línea y los saltos de línea repetidos.
3. `fragmentar()` aplica `RecursiveCharacterTextSplitter` con `chunk_size=500` y `chunk_overlap=50`, medidos en tokens reales con `tiktoken`. El splitter corta primero por párrafos, después por líneas y por último por espacios, para no partir una idea por la mitad.
4. `ingest()` guarda los fragmentos en una colección persistente de ChromaDB con distancia coseno.

Con el dataset incluido se generan 6 fragmentos de entre 422 y 494 tokens, dos por documento.

Tres decisiones de la ingesta:

- **No se reindexa si la base ya existe.** `ingest()` consulta primero si la colección tiene fragmentos y, si los tiene, termina. `python ingest.py --force` borra la colección y la vuelve a crear.
- **Ids deterministas.** Cada fragmento se identifica con `archivo:posición:hash del contenido`, así que volver a ingerir el mismo contenido actualiza los registros en lugar de duplicarlos.
- **Un único modelo de embeddings.** `build_embeddings()` es el único lugar donde se crea el modelo, y lo usan tanto la ingesta como la consulta. Además, el nombre de la colección incluye el nombre del modelo (por ejemplo `atlas_text_embedding_3_small`), de modo que nunca se consulta con un modelo una colección indexada con otro.

Hay dos modelos de embeddings disponibles: `text-embedding-3-small` de OpenAI y, para trabajar sin clave, `all-MiniLM-L6-v2`, que ChromaDB ejecuta en forma local.

### Capa de recuperación y cadena LCEL (`rag.py`)

```python
generacion = (
    RunnablePassthrough.assign(contexto=lambda x: format_docs(x["documentos"]))
    | prompt
    | model
    | parser
).with_retry(retry_if_exception_type=(OutputParserException,), stop_after_attempt=2)

chain = {"documentos": retriever, "pregunta": RunnablePassthrough()} | RunnablePassthrough.assign(
    respuesta=generacion
)
```

- `retriever` es `vectorstore.as_retriever(search_kwargs={"k": 4})`: convierte la pregunta en un embedding y trae los 4 fragmentos más cercanos. El `top_k` se mantiene entre 3 y 5 para no diluir la atención del modelo.
- `format_docs` es el transformador de documentos: une los fragmentos y antepone a cada uno su fuente, `[fuente: despliegues.md]`.
- `parser` es un `PydanticOutputParser` sobre `RAGResponse`. Sus instrucciones de formato entran al prompt como variable.
- Los documentos recuperados viajan junto a la respuesta para poder verificar las referencias después.

### Prompt de sistema

Funciona como filtro de veracidad:

> Eres un asistente técnico. Responde solo basándote en el CONTEXTO proporcionado. Si la respuesta no está allí, responde exactamente "No lo sé" y deja las referencias vacías. No uses conocimiento propio ni completes lo que falte.

### `get_rag_response(query: str)`

Es una corrutina que cumple los cuatro pasos de la consigna:

1. Busca por similitud en ChromaDB.
2. Construye el prompt con los fragmentos recuperados.
3. Llama al modelo de forma asíncrona, con `await chain.ainvoke(query)`.
4. Parsea la respuesta al modelo Pydantic `RAGResponse`, con `texto` y `referencias`.

Después aplica dos controles propios:

- **Referencias verificadas.** Se descarta cualquier referencia que el modelo cite y que no corresponda a un fragmento recuperado.
- **"No lo sé" sin referencias.** Si el modelo responde que no sabe, la lista de referencias queda vacía.

Si el modelo no devuelve el formato pedido se reintenta una vez. Ante un fallo (formato, clave, red), la función devuelve `None` y deja el motivo en el log, sin interrumpir el programa.

### Modelo y `temperature`

`ChatOpenAI` y `ChatAnthropic` se crean con `temperature=0`, para que la respuesta se apegue al contexto. El modelo por defecto de Anthropic es `claude-haiku-4-5` porque acepta ese parámetro; los modelos Claude más recientes (por ejemplo `claude-opus-5-5`) ya no lo usan y lo rechazan con un error 400.

## Las dos pruebas

`main.py` hace dos preguntas:

| Prueba | Pregunta | Resultado esperado |
|---|---|---|
| Con respuesta | ¿Cuántas réplicas mínimas debe tener un servicio crítico en producción? | Un mínimo de 3 réplicas, con referencia a `despliegues.md` |
| Trampa | ¿Cuál es el presupuesto anual de la plataforma Atlas? | "No lo sé", sin referencias: el dato no figura en ningún documento |

Forma de la salida en cada caso (el texto exacto de la primera varía según el modelo):

```json
{
  "texto": "Los servicios críticos (gateway, pagos y antifraude) deben correr con un mínimo de 3 réplicas.",
  "referencias": ["despliegues.md"]
}
```

```json
{
  "texto": "No lo sé",
  "referencias": []
}
```

Esta es la salida de `python main.py` para la primera prueba, sin clave de un modelo de lenguaje y con embeddings locales. La ingesta y la recuperación funcionan; la generación informa que falta la clave y el programa termina sin error:

```
INFO    Ingesta lista: 3 documentos, 6 fragmentos, embeddings all-MiniLM-L6-v2

=== Pregunta con respuesta en los documentos ===
¿Cuántas réplicas mínimas debe tener un servicio crítico en producción?

Fragmentos recuperados:
  - despliegues.md (fragmento 1): ## Ventanas de despliegue Los despliegues a producción se hacen de lun...
  - despliegues.md (fragmento 0): # Plataforma Atlas: despliegues ## Entornos Atlas tiene tres entornos....
  - incidentes.md (fragmento 0): # Plataforma Atlas: gestión de incidentes ## Niveles de severidad Los ...
  - incidentes.md (fragmento 1): La prioridad es restablecer el servicio, no encontrar la causa. Si un ...

ERROR   Falta la variable de entorno OPENAI_API_KEY
Respuesta:
Sin respuesta: el motivo está en el log.
```

El primer fragmento recuperado es el que contiene el dato pedido: la sección "Requisitos para producción" de `despliegues.md`.

## Tests

`test_main.py` tiene 24 tests. No usan la red ni claves: los embeddings se reemplazan por `DeterministicFakeEmbedding` de LangChain, el modelo por un doble que devuelve respuestas preparadas, y la base se crea en una carpeta temporal.

```bash
pytest -v
```

Cubren los cuatro criterios de la rúbrica:

- **Chunking e ingesta:** carga desde `data/`, límite de 500 tokens por fragmento, solapamiento, ids deterministas, persistencia en disco, no reindexar si la base existe, una colección por modelo de embeddings.
- **Arquitectura asíncrona y recuperación:** el retriever devuelve `top_k` fragmentos; `get_rag_response` entrega al modelo el contexto recuperado y devuelve texto y referencias.
- **Validación con Pydantic:** `RAGResponse` valida sus campos; una salida mal formada se reintenta y, si persiste, se informa.
- **Robustez:** la pregunta trampa termina en "No lo sé" sin referencias; las referencias inventadas se descartan; sin clave o con fallas de red no hay excepción; las claves no están en el repositorio.

## Reporte de tests

Salida de `pytest -v`:

```
============================= test session starts ==============================
collecting ... collected 24 items

test_main.py::test_carga_los_documentos_de_data_con_su_fuente PASSED     [  4%]
test_main.py::test_limpiar_quita_espacios_finales_y_saltos_repetidos PASSED [  8%]
test_main.py::test_los_fragmentos_respetan_el_tamano_en_tokens_y_se_solapan PASSED [ 12%]
test_main.py::test_los_ids_son_deterministas_y_unicos PASSED             [ 16%]
test_main.py::test_ingest_persiste_en_disco_y_no_reindexa_si_ya_existe PASSED [ 20%]
test_main.py::test_ingest_con_force_reindexa_sin_duplicar PASSED         [ 25%]
test_main.py::test_ingest_sin_archivos_no_falla PASSED                   [ 29%]
test_main.py::test_cada_modelo_de_embeddings_tiene_su_propia_coleccion PASSED [ 33%]
test_main.py::test_sin_clave_de_openai_los_embeddings_son_locales PASSED [ 37%]
test_main.py::test_el_retriever_devuelve_top_k_fragmentos PASSED         [ 41%]
test_main.py::test_el_prompt_exige_responder_solo_con_el_contexto PASSED [ 45%]
test_main.py::test_format_docs_incluye_la_fuente_de_cada_fragmento PASSED [ 50%]
test_main.py::test_get_rag_response_devuelve_texto_y_referencias PASSED  [ 54%]
test_main.py::test_rag_response_valida_texto_y_referencias PASSED        [ 58%]
test_main.py::test_reintenta_si_la_salida_no_cumple_el_esquema PASSED    [ 62%]
test_main.py::test_devuelve_none_si_el_modelo_nunca_cumple_el_formato PASSED [ 66%]
test_main.py::test_pregunta_trampa_responde_no_lo_se_sin_referencias PASSED [ 70%]
test_main.py::test_descarta_referencias_que_no_estaban_en_el_contexto PASSED [ 75%]
test_main.py::test_devuelve_none_si_falla_la_api PASSED                  [ 79%]
test_main.py::test_rechaza_una_consulta_vacia[] PASSED                   [ 83%]
test_main.py::test_rechaza_una_consulta_vacia[   ] PASSED                [ 87%]
test_main.py::test_sin_argumentos_ni_clave_no_se_rompe PASSED            [ 91%]
test_main.py::test_las_claves_no_estan_en_el_repositorio PASSED          [ 95%]
test_main.py::test_main_corre_las_dos_pruebas_sin_clave PASSED           [100%]

============================== 24 passed in 4.41s ==============================
```
