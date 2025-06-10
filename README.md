# Health-Chatbot
CURA is an AI-powered healthcare chatbot that combines Retrieval-Augmented Generation (RAG) with OCR capabilities to deliver accurate, real-time medical information and nutritional label analysis. Designed with responsiveness, scalability, and reliability in mind, CURA leverages cutting-edge tools like Groq-hosted LLaMA 3, Pinecone vector database, HuggingFace embeddings, and Tesseract OCR.

**Project Highlights**
Medical Chat Mode: Uses a RAG pipeline to answer health-related questions using information sourced from The Gale Encyclopedia of Medicine.
Nutrition Label Decoder: Accepts uploaded food label images, extracts text using OCR, and provides insights on nutritional content.
Low Latency: Achieves average response times 75% faster than GPT-3.5 Turbo using the Groq API with LLaMA 3 8B.
Modular Design: Built with Flask for easy integration, independent module use, and scalability.

**System Architecture**
Retrieval-Augmented Generation Pipeline:
Embedding Generation: Text chunks from medical PDFs are embedded using HuggingFace models.
Vector Storage: Embeddings stored in Pinecone for fast Approximate Nearest Neighbor (ANN) search.
Contextual Prompting: Top relevant chunks are merged with user queries.
Answer Generation: Prompts are passed to Groq-hosted LLaMA 3 8B for low-latency, accurate answers.

Nutrition Label Decoder Pipeline:
Images are processed using Pillow and Tesseract OCR.
Text is interpreted and analyzed using a lightweight RAG approach with nutrition heuristics.

**Tech Stack**
Backend: Python, Flask
LLM: Groq-hosted LLaMA 3 8B
Embedding & Chunking: HuggingFace Transformers, LangChain
Vector DB: Pinecone
OCR: Tesseract, Pillow
Knowledge Base: Gale Encyclopedia of Medicine (PDF)



