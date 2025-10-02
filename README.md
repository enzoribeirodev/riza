# RIZA - RAG Chat Interface

Uma solução genérica e modular para sistemas RAG (Retrieval Augmented Generation) com foco em reprodutibilidade, personalização e eficiência. RIZA visa ser uma plataforma extensível para conversação com documentos, integrando múltiplas tecnologias e provedores de LLM.

---

## 📋 Índice

- [Visão Geral](#🎯-visão-geral)
- [Arquitetura](#🏗️-arquitetura)
- [Estrutura do Projeto](#📁-estrutura-do-projeto)
- [Funcionalidades](#✨-funcionalidades)
- [Instalação](#🚀-instalação)
- [Como Usar](#💻-como-usar)
- [Stack Tecnológica](#🛠️-stack-tecnológica)
- [Pipeline RAG](#🔄-pipeline-rag)
- [Decisões Técnicas](#🎯-decisões-técnicas)
- [Configuração Avançada](#⚙️-configuração-avançada)
- [Próximos Passos](#🚧-próximos-passos)

---

## 🎯 Visão Geral

RIZA é uma prova de conceito (PoC) de um assistente de perguntas e respostas que utiliza documentos como fonte de verdade. O projeto demonstra a implementação completa de um pipeline RAG, desde a ingestão de documentos até a geração de respostas contextualizadas, com suporte a múltiplos provedores de LLM e modelos de embedding.

### Motivação

Inicialmente desenvolvido para Q&A sobre manuais técnicos (testada inicialmente com o manual do Samsung Galaxy Z Flip 7), o projeto evoluiu para uma solução genérica aplicável a diversos domínios, incluindo:

- Assistência técnica e suporte
- Análise de documentos jurídicos
- Leitura e síntese de artigos científicos
- Documentação corporativa
- Bases de conhecimento internas

### Diferencial

RIZA não é apenas mais um RAG básico. O projeto implementa:

- **Compression Retriever com Reranking**: Melhora significativa na qualidade dos documentos recuperados
- **Chunking híbrido inteligente**: Baseado em estrutura de markdown com contexto preservado
- **Limpeza robusta**: Remoção de artefatos de conversão (tags HTML, imagens, etc)
- **Multi-provider**: Suporte nativo para Ollama, OpenAI e Google Gemini
- **Embeddings flexíveis**: HuggingFace (local) e Ollama
- **Modularidade**: Código organizado em módulos reutilizáveis

---

## 🏗️ Arquitetura

```
┌─────────────────┐
│   PDF/TXT       │
│   Document      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Docling       │
│   (PDF → MD)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Chunking      │
│   (Hybrid)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Cleaning      │
│   (Regex)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Embeddings    │
│ (HF or Ollama)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ChromaDB      │
│  (Vector Store) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Retrieval     │
│   (base_k=10)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Reranking     │
│ (CrossEncoder)  │
│   (top_n=4)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│      LLM        │
│ (Ollama/API)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Response     │
│   + Sources     │
└─────────────────┘
```

---

## 📁 Estrutura do Projeto

```
riza/
├── app.py                          # Interface Streamlit principal
├── Modelfile                       # Configuração customizada para Ollama
├── .env.example                    # Template de variáveis de ambiente
├── requirements.txt                # Dependências Python
├── README.md                       # Esta documentação
│
├── data/
│   ├── raw/                        # Documentos fonte
│   │   ├── galaxy_z_flip_7.pdf    # Manual de teste completo
│   │   └── galaxy_z_flip_7_teste.pdf  # Versão reduzida para testes
│   │
│   └── processed/                  # Dados processados
│       ├── manual_teste_markdown.md    # Conversão intermediária
│       └── chroma_db/              # Vector stores persistidos
│           └── [hash]/             # Coleção por documento
│
├── notebooks/                      # Exploração e prototipagem
│   ├── 01_data_ingestion.ipynb    # Teste de conversão PDF → Markdown
│   ├── 02_embedding_indexing.ipynb    # Experimentos com embeddings
│   └── 03_rag_pipeline.ipynb      # Desenvolvimento do pipeline RAG
│
└── src/                            # Código de produção
    ├── __init__.py
    ├── data_processing.py          # Processamento e limpeza de documentos
    ├── vector_store.py             # Gerenciamento do ChromaDB
    └── rag_components.py           # LLM, embeddings e retrieval
```

### Organização

- **`notebooks/`**: Ambiente de experimentação onde foram prototipados os primeiros modelos e testadas diferentes abordagens. Cada notebook representa uma etapa do desenvolvimento do pipeline.

- **`src/`**: Código final modularizado e otimizado. Contém as implementações definitivas das funções utilizadas no sistema.

- **`data/raw/`**: Armazena o manual do Galaxy Z Flip 7, usado inicialmente como caso de teste para validar o sistema de Q&A.

---

## 🔧 Funcionalidades

### Implementadas

1. **Ingestão de Documentos**
   - Suporte a PDF via Docling
   - Conversão inteligente para Markdown preservando estrutura
   - Processamento de headers e seções

2. **Chunking Híbrido**
   - Divisão baseada em estrutura (headers H2 e H3, por padrão, permitindo configuração)
   - Chunk size configurável (padrão: 600 tokens)
   - Overlap inteligente (padrão: 100 tokens)
   - Preservação de contexto hierárquico

3. **Limpeza Robusta**
   - Remoção de tags HTML
   - Eliminação de comentários (`<!-- image -->`, etc)
   - Normalização de espaços e quebras de linha
   - Filtragem por tamanho mínimo

4. **Embeddings Flexíveis**
   - **HuggingFace** (local): `all-MiniLM-L6-v2` (padrão)
   - **Ollama**: `nomic-embed-text`
   - Fácil extensão para outros modelos

5. **Indexação Vetorial**
   - ChromaDB com persistência local
   - Collections isoladas por documento (evita interferência)
   - Reutilização automática de vector stores existentes

6. **Retrieval Inteligente**
   - Base retrieval: top-k documentos (configurável: 5-20)
   - **Compression Retriever**: Reranking com CrossEncoder
   - Modelo: `cross-encoder/ms-marco-MiniLM-L-6-v2`
   - Top-n após reranking (configurável: 1-10)

7. **Múltiplos Provedores LLM**
   - **Ollama** (local): phi3:mini, llama3.2, mistral, etc
   - **OpenAI API**: GPT-3.5, GPT-4, GPT-4o
   - **Google Gemini API**: Gemini 2.5 Flash/Pro

8. **Interface Web**
   - Streamlit com design limpo
   - Upload de documentos
   - Configuração de modelos em tempo real
   - Histórico de conversação
   - Visualização de fontes utilizadas

9. **Rastreabilidade**
   - Citação de trechos fonte
   - Indicação de relevância
   - Transparência na origem das respostas

### Trade-offs e Decisões

| Aspecto | Decisão | Justificativa |
|---------|---------|---------------|
| **Conversão PDF** | Docling | Melhor preservação de estrutura vs PyPDF |
| **Chunking** | Híbrido (headers + size) | Balanceia contexto semântico e tamanho |
| **Vector DB** | ChromaDB | Local, leve, suficiente para PoC |
| **Embedding padrão** | HuggingFace | Execução local, zero custo |
| **Reranking** | CrossEncoder | Melhoria significativa na relevância |
| **LLM padrão** | Ollama local | Privacidade, custo zero, reprodutível |

---

## 🚀 Instalação

### Pré-requisitos

- Python 3.10+
- 8GB+ RAM
- (Opcional) GPU para modelos locais maiores

### Passo 1: Clone o repositório

```bash
git clone https://github.com/seu-usuario/riza.git
cd riza
```

### Passo 2: Crie ambiente virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### Passo 3: Instale dependências

```bash
pip install -r requirements.txt
```

### Passo 4: Configure Ollama (opcional)

Se quiser usar modelos locais:

```bash
# Instalar Ollama
curl -fsSL https://ollama.com/install.sh | sh  # Linux/Mac

# Baixar modelos
ollama pull phi3:mini
ollama pull nomic-embed-text
```

### Passo 5: Configure APIs (opcional)

Se quiser usar OpenAI ou Google:

```bash
cp .env.example .env
# Edite .env e adicione suas chaves:
# OPENAI_API_KEY=...
# GOOGLE_API_KEY=...
```

---

## 💻 Como Usar

### Interface Web (Recomendado)

```bash
streamlit run app.py
```

Acesse `http://localhost:8501` e:

1. **Configure modelos** na sidebar:
   - Embedding: HuggingFace ou Ollama
   - LLM: Ollama, OpenAI ou Google
   - Ajuste temperature e parâmetros de retrieval

2. **Faça upload de um PDF**

3. **Aguarde processamento** (primeira vez: ~2-5 min)

4. **Converse!** Faça perguntas sobre o documento

### Programático

```python
from src.data_processing import process_pdf_to_chunks
from src.vector_store import build_vector_store
from src.rag_components import (
    create_embedding_model,
    create_llm,
    create_compression_retriever,
    create_rag_chain,
    query_rag
)

# 1. Processar documento
chunks = process_pdf_to_chunks("data/raw/seu_documento.pdf")

# 2. Criar embeddings
embedding_model = create_embedding_model("huggingface", "all-MiniLM-L6-v2")

# 3. Indexar
vector_store = build_vector_store(
    chunks, 
    embedding_model, 
    "data/processed/chroma_db",
    collection_name="meu_doc"
)

# 4. Criar retriever com reranking
retriever = create_compression_retriever(vector_store, base_k=10, top_n=4)

# 5. Criar LLM
llm = create_llm("ollama", "phi3:mini", temperature=0.7)

# 6. Criar chain RAG
qa_chain = create_rag_chain(llm, retriever)

# 7. Fazer perguntas
response = query_rag(qa_chain, "Qual a capacidade da bateria?")
print(response["result"])
print(response["source_documents"])
```

---

## 🛠️ Stack Tecnológica

### Bibliotecas Principais

| Componente | Biblioteca | Versão | Função |
|------------|-----------|--------|--------|
| PDF Processing | Docling | 1.0+ | Conversão PDF → Markdown |
| Text Splitting | LangChain | 0.3+ | Chunking inteligente |
| Embeddings | HuggingFace | 0.1+ | Geração de vetores (local) |
| | Ollama | 0.2+ | Embeddings via API local |
| Vector DB | ChromaDB | 0.4+ | Armazenamento vetorial |
| Reranking | HuggingFace CrossEncoder | - | Reordenação de resultados |
| LLM | LangChain Ollama | 0.2+ | Interface para modelos locais |
| | LangChain OpenAI | 0.2+ | Interface para GPT |
| | LangChain Google | 2.0+ | Interface para Gemini |
| Interface | Streamlit | 1.28+ | Web UI |
| Utilities | python-dotenv | 1.0+ | Gerenciamento de variáveis |

### Modelos

**Embeddings:**
- `all-MiniLM-L6-v2` (HuggingFace) - 384 dim, 22M params
- `nomic-embed-text` (Ollama) - 768 dim, otimizado para RAG

**Reranking:**
- `cross-encoder/ms-marco-MiniLM-L-6-v2` - Treinado em MS MARCO

**LLM (Ollama):**
- `phi3:mini` - 3.8B params, rápido
- `llama3.2` - 3B params, balanceado
- `mistral` - 7B params, qualidade

**LLM (APIs):**
- OpenAI: GPT-3.5, GPT-4, GPT-4o
- Google: Gemini Pro, Gemini 1.5 Flash/Pro

---

## 🔄 Pipeline RAG

### 1. Pré-processamento

```python
# data_processing.py

def load_and_convert_to_markdown(file_path):
    # Converte PDF para Markdown usando Docling
    # Preserva estrutura de headers, parágrafos, listas
    pass

def chunk_text(markdown_text, chunk_size=600, chunk_overlap=100):
    # Chunking híbrido:
    # 1. Divide por headers (H2, H3)
    # 2. Adiciona contexto hierárquico como prefixo
    # 3. Subdivide chunks grandes recursivamente
    pass

def clean_and_filter_chunks(chunks, min_length=50):
    # Limpeza:
    # - Remove tags HTML
    # - Remove comentários (<!-- image -->)
    # - Normaliza espaços
    # - Filtra chunks muito pequenos
    pass
```

**Parâmetros configuráveis:**
- `chunk_size`: 400-800 (padrão: 600)
- `chunk_overlap`: 50-150 (padrão: 100)
- `min_length`: 30-100 (padrão: 50)

### 2. Indexação

```python
# vector_store.py

def build_vector_store(chunks, embedding_model, persist_directory, collection_name):
    # Cria collection única por documento (hash MD5)
    # Gera embeddings para cada chunk
    # Persiste no ChromaDB
    # Permite reutilização em execuções futuras
    pass
```

**Otimizações:**
- Collections isoladas (evita cross-contamination)
- Persistência automática
- Detecção de vector stores existentes

### 3. Retrieval + Reranking

```python
# rag_components.py

def create_compression_retriever(vector_store, base_k=10, top_n=4):
    # Pipeline de 2 estágios:
    # Stage 1: Retrieval por similaridade de embeddings (base_k)
    # Stage 2: Reranking com CrossEncoder (top_n)
    pass
```


### 4. Geração

```python
def create_rag_chain(llm, retriever, chain_type="stuff"):
    # Monta chain LangChain com:
    # - Retriever configurado
    # - LLM escolhido
    # - Return source documents
    pass

def query_rag(qa_chain, question):
    # Executa query usando .invoke()
    # Retorna resposta + documentos fonte
    pass
```

---

## 🎯 Decisões Técnicas

### 1. Por que Docling?

**Alternativas avaliadas:**
- PyPDF2: Perde formatação
- PyMuPDF: Bom mas complexo
- pdfplumber: Focado em tabelas
- **Docling**: Melhor preservação de estrutura Markdown

**Vantagens:**
- Mantém hierarquia de headers
- Detecta listas e tabelas
- Conversão limpa

### 2. Por que ChromaDB?

**Alternativas avaliadas:**
- Zero configuração
- Persistência automática
- Isolamento por collection

### 3. Por que Compression Retriever?

**Problema:** Similarity search é rápido, mas não tão preciso

**Solução:** Two-stage retrieval
1. Top K (base_k=10)
2. Rerank with better model (top_n=4)

### 4. Por que múltiplos provedores?

**Flexibilidade:**
- **Ollama**: Dev local, privacidade, zero custo
- **OpenAI**: Máxima qualidade
- **Google**: Bom custo-benefício

### 5. Segurança e Privacidade

**Dados sensíveis:**
- Usar **apenas Ollama** (tudo local)
- Vector stores em `data/processed/`
- Nunca sai da máquina

**APIs externas:**
- Dados trafegam para OpenAI/Google
- Avaliar compliance (LGPD, GDPR)
- Revisar termos de serviço

---

## ⚙️ Configuração Avançada

### Customização via Modelfile (Ollama)

O arquivo `Modelfile` na raiz do projeto permite customizar o comportamento dos modelos Ollama:

```dockerfile
FROM {MODEL_NAME}

TEMPLATE """<|system|>
You are a {TYPE FUNC HERE} assistant specialized in {DOMAIN HERE}.
Your responses should be {STYLE HERE} and always {CONSTRAINT HERE}.<|end|>
<|user|>
{{ .Prompt }}<|end|>
<|assistant|>
"""

PARAMETER temperature {TEMPERATURE}
PARAMETER top_p {TOP_P}
PARAMETER stop "<|end|>"
PARAMETER stop "<|endoftext|>"

SYSTEM """You are {ROLE HERE}. {ADDITIONAL INSTRUCTIONS HERE}."""
```

**Exemplos de uso:**

**Assistente Técnico:**
```dockerfile
FROM phi3:mini

TEMPLATE """<|system|>
You are a technical support assistant specialized in smartphone troubleshooting.
Your responses should be clear, step-by-step and always reference the manual.<|end|>
<|user|>
{{ .Prompt }}<|end|>
<|assistant|>
"""

PARAMETER temperature 0.3
PARAMETER top_p 0.9

SYSTEM """You are a technical support specialist. Always provide safety warnings when relevant."""
```

**Assistente Jurídico:**
```dockerfile
FROM llama3.2

TEMPLATE """<|system|>
You are a legal document analyst specialized in contract review.
Your responses should be formal, precise and always cite specific clauses.<|end|>
<|user|>
{{ .Prompt }}<|end|>
<|assistant|>
"""

PARAMETER temperature 0.1
PARAMETER top_p 0.85
```

**Para usar:**
```bash
ollama create assistente-tecnico -f Modelfile
# Use "assistente-tecnico" no RIZA
```

### Ajuste de Parâmetros (Recomendação)

**Chunking para documentos técnicos:**
```python
chunks = process_pdf_to_chunks(
    file_path,
    chunk_size=800,      # Chunks maiores
    chunk_overlap=150,   # Mais overlap
    min_length=100       # Chunks mais substanciais
)
```

**Retrieval para documentos longos:**
```python
retriever = create_compression_retriever(
    vector_store,
    base_k=15,    # Busca mais ampla
    top_n=7       # Mais contexto
)
```

**LLM para respostas criativas:**
```python
llm = create_llm("ollama", "mistral", temperature=0.9)
```

---

## 🚧 Próximos Passos

### Curto Prazo

1. **Fine-tuning com Unsloth/Axolotl**
   - Treinar phi3:mini em domínios específicos
   - LoRA para customização de estilo/terminologia
   - Útil para: artigos científicos, documentos jurídicos
   - Implementação: Google Colab (GPU grátis)

2. **Embeddings via API**
   - OpenAI: `text-embedding-3-small/large`
   - Cohere: `embed-multilingual-v3`
   - Voyage AI: `voyage-2`
   - Comparação de qualidade/custo

3. **Mais Provedores LLM**
   - Anthropic Claude (via Bedrock ou direto)
   - Mistral API
   - Groq (inferência ultra-rápida)
   - Together AI (modelos open-source)

### Médio Prazo

4. **RAG Multimodal Robusto**
   - Extração de imagens, diagramas, tabelas
   - Descrição automática de imagens (BLIP, LLaVA)
   - Indexação conjunta texto+imagens
   - OCR para documentos escaneados

5. **Interface Avançada**
   - Editor de chunks pré-indexação
   - Comparação visual de embeddings (t-SNE/UMAP)
   - Ajuste de parâmetros em tempo real
   - Histórico persistente de conversas
   - Export de resultados (PDF, Markdown)

6. **Avaliação Automática**
   - Dataset de Q&A ground-truth
   - Métricas: ROUGE, BLEU, BERTScore
   - Similarity entre resposta e fonte
   - A/B testing entre configurações

### Longo Prazo

7. **Integração Web Search**
   - Fallback quando documento não tem resposta
   - APIs: Brave Search, SerpAPI, Tavily
   - Fusão RAG local + web results

8. **Agentic RAG**
   - ReAct pattern (reason + act)
   - Tool use (calculator, API calls)
   - Multi-step reasoning

9. **Produtização**
   - API FastAPI
   - Containerização (Docker)
   - CI/CD pipeline
   - Monitoring e logging (MLflow, W&B)
   - Rate limiting e autenticação

---