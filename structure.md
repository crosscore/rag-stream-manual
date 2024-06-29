rag-streaming/
├── frontend/
│   ├── main.py
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── static/
│   │   └── styles.css
│   └── templates/
│       └── index.html
│
├── backend/
│   ├── main.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── utils/
│       └── vectorizer.py
│
├── pgvector_manual/
│   ├── Dockerfile
│   └── init_pgvector.sql
│
├── s3_db/
│   ├── main.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── data/
│       ├── pdf/
│       ├── xlsx/
│       └── docx/
│
├── docker-compose.yml
└── .env
