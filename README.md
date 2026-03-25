# Classificador de Veículos 🚗

Aplicação de visão computacional que classifica veículos em imagens usando **YOLOv8** (Ultralytics).

## Funcionalidades

- Upload de imagem ou captura direta pela câmera
- Detecção e classificação de veículos (Carro, Caminhão, Ônibus, Motocicleta)
- Imagem anotada com *bounding boxes* e rótulos de confiança
- Interface web construída com **Streamlit**
- API assíncrona construída com **FastAPI**

---

## Estrutura do Projeto

```
ambar/
├── backend/
│   ├── main.py           # API FastAPI com inferência YOLOv8
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app.py            # Interface Streamlit
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml    # Orquestração dos dois serviços
└── README.md
```

---

## Pré-requisitos

- **Docker + Docker Compose** (recomendado), **ou**
- Python 3.10+ com `pip`

---

## Execução com Docker (recomendado)

```bash
docker-compose up --build
```

Após a inicialização:
| Serviço  | URL                    |
|----------|------------------------|
| Frontend | http://localhost:8501  |
| Backend  | http://localhost:8000  |
| API Docs | http://localhost:8000/docs |

> Na primeira execução o YOLOv8n (~6 MB) é baixado automaticamente.

---

## Execução manual (sem Docker)

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

---

## Endpoints da API

| Método | Rota        | Descrição                                     |
|--------|-------------|-----------------------------------------------|
| GET    | `/health`   | Verifica status do serviço e do modelo        |
| POST   | `/classify` | Recebe imagem e retorna classificação + imagem anotada |

### Exemplo de resposta do `/classify`

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

## Classes de Veículos Suportadas

| Classe COCO | Rótulo      |
|-------------|-------------|
| 2           | Carro       |
| 3           | Motocicleta |
| 5           | Ônibus      |
| 7           | Caminhão    |

---

## Stack Tecnológica

| Camada    | Tecnologia                          |
|-----------|-------------------------------------|
| Backend   | Python 3.11, FastAPI, Uvicorn       |
| Modelo IA | YOLOv8n (Ultralytics, COCO)         |
| Visão     | OpenCV, Pillow, NumPy               |
| Frontend  | Streamlit                           |
| DevOps    | Docker, Docker Compose              |
