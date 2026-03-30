# AGENTS.md — Instruções para Agentes e Desenvolvedores

Guia de referência rápida para qualquer agente de IA ou desenvolvedor que trabalhe neste repositório.

---

## Visão Geral do Projeto

**Classificador de Veículos** é uma aplicação web de visão computacional que detecta e classifica veículos (Carro, Motocicleta, Ônibus, Caminhão) em imagens usando o modelo **YOLOv8n** da Ultralytics.

| Camada    | Tecnologia                       | Porta  |
|-----------|----------------------------------|--------|
| Backend   | Python 3.11 · FastAPI · Uvicorn  | 8000   |
| Frontend  | Python 3.11 · Streamlit          | 8501   |
| Modelo IA | YOLOv8n (Ultralytics · COCO)     | —      |
| DevOps    | Docker · Docker Compose          | —      |

---

## Estrutura do Repositório

```
ambar/
├── backend/
│   ├── main.py           # API FastAPI — inferência YOLOv8, endpoints /health e /classify
│   ├── requirements.txt  # Dependências Python do backend
│   └── Dockerfile
├── frontend/
│   ├── app.py            # Interface Streamlit — upload/câmera, exibição de resultados
│   ├── requirements.txt  # Dependências Python do frontend
│   └── Dockerfile
├── docker-compose.yml    # Orquestração dos dois serviços
├── README.md
└── AGENTS.md             # Este arquivo
```

---

## Como Executar

### Com Docker (recomendado)

```bash
docker-compose up --build
```

> Na primeira execução o YOLOv8n (~6 MB) é baixado automaticamente e cacheado no volume `model_cache`.

### Sem Docker (desenvolvimento local)

```bash
# Terminal 1 — Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Frontend
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

---

## Variáveis de Ambiente

| Variável          | Serviço  | Padrão       | Descrição                                                      |
|-------------------|----------|--------------|----------------------------------------------------------------|
| `MODEL_WEIGHTS`   | backend  | `yolov8n.pt` | Caminho ou nome do arquivo de pesos YOLOv8                     |
| `ALLOWED_ORIGINS` | backend  | `*`          | Origens CORS permitidas (separar por vírgula em produção)      |
| `MAX_IMAGE_PIXELS`| backend  | `25000000`   | Limite de pixels para proteção contra decompression bomb       |
| `API_URL`         | frontend | `http://localhost:8000` | URL base da API usada pelo Streamlit              |

---

## API Endpoints

| Método | Rota        | Descrição                                               |
|--------|-------------|---------------------------------------------------------|
| GET    | `/health`   | Retorna `{"status": "ready"}` quando o modelo está carregado |
| POST   | `/classify` | Recebe uma imagem multipart e retorna detecções + imagem anotada em base64 |

### Resposta de `/classify`

```json
{
  "detections": [
    {
      "label": "Carro",
      "confidence": 0.92,
      "box": [120, 80, 640, 400],
      "class_id": 2
    }
  ],
  "annotated_image": "<base64-encoded JPEG>",
  "vehicles_found": true
}
```

---

## Classes de Veículos Detectadas

| COCO class_id | Rótulo PT   |
|---------------|-------------|
| 2             | Carro       |
| 3             | Motocicleta |
| 5             | Ônibus      |
| 7             | Caminhão    |

Detecções com confiança abaixo de **0.25** são descartadas (`CONFIDENCE_THRESHOLD`).

---

## Convenções de Código

### Backend (`backend/main.py`)

- **Linguagem**: Python 3.11+, tipagem estrita com `from __future__ import annotations`.
- **Framework**: FastAPI com lifespan para carregar o modelo uma única vez na inicialização.
- **Modelo**: instância única em `ModelContainer` (singleton global).
- **Validações**: arquivo e tamanho são validados em `validate_upload()` **antes** da decodificação; proteção contra decompression bomb em `decode_image()`.
- **Erros**: sempre elevar `HTTPException` com status codes semânticos (413, 415, 422, 503).
- **Logs**: usar `logger = logging.getLogger(__name__)` em vez de `print()`.
- **Constantes**: definidas no topo do arquivo em UPPER_SNAKE_CASE; configurações de ambiente lidas via `os.getenv()`.

### Frontend (`frontend/app.py`)

- **Framework**: Streamlit; toda a UI encapsulada em funções `render_*`.
- **Comunicação**: `requests` síncrono; erros de rede capturados e convertidos em `RuntimeError` com mensagens em português.
- **Configuração**: a URL da API é lida de `API_URL` (env var), com fallback para `http://localhost:8000`.

### Geral

- Strings voltadas ao usuário em **português (BR)**.
- Comentários e docstrings de código em **inglês**.
- Sem frameworks de teste configurados por enquanto — adicionar `pytest` ao `requirements.txt` antes de criar testes.

---

## Guia para Agentes de IA

### Antes de alterar qualquer arquivo

1. Leia este `AGENTS.md` inteiro.
2. Leia os arquivos relevantes (`main.py`, `app.py`) antes de propor mudanças.
3. Verifique se a mudança afeta o contrato da API (campos do JSON de resposta, status codes) — qualquer alteração nesse contrato impacta o frontend.

### Regras obrigatórias

- **Não remova** as validações de segurança em `validate_upload()` e `decode_image()`.
- **Não mude** a estrutura do JSON de resposta do `/classify` sem atualizar o frontend.
- **Não use** `print()` no backend; use sempre o `logger`.
- **Não adicione** dependências sem atualizar o `requirements.txt` correspondente **e** o `Dockerfile`.
- **Mantenha** a compatibilidade com Docker: variáveis de ambiente devem ter valores padrão sensatos.

### Adicionando novos endpoints

1. Implemente a lógica de validação como função helper separada.
2. Documente o endpoint com docstring (usado pelo Swagger em `/docs`).
3. Use `async def` para endpoints I/O-bound; funções CPU-bound (inferência) podem executar síncronamente ou via `run_in_executor`.

### Adicionando novas classes de veículos

1. Adicione o `class_id` COCO em `VEHICLE_CLASS_IDS` (backend).
2. Adicione o rótulo em `VEHICLE_LABELS` (backend).
3. Atualize a tabela de classes no `README.md` e neste arquivo.

### Mudando o modelo YOLOv8

- Use a variável de ambiente `MODEL_WEIGHTS` para apontar para outro arquivo de pesos.
- Modelos maiores (`yolov8s.pt`, `yolov8m.pt`) requerem mais RAM e CPU/GPU.
- Atualize `start_period` no healthcheck do `docker-compose.yml` se o carregamento demorar mais.

---

## Checklist de Pull Request

- [ ] Código segue as convenções descritas acima
- [ ] Variáveis de ambiente novas têm valor padrão e estão documentadas neste arquivo
- [ ] `requirements.txt` atualizado se novas dependências foram adicionadas
- [ ] Contrato da API preservado (ou frontend atualizado em conjunto)
- [ ] README.md atualizado se funcionalidades visíveis ao usuário mudaram
