# 🏥 Medical Chatbot

A conversational AI assistant for medical information, powered by The Gale Encyclopedia of Medicine, advanced language models (LLMs), and semantic search. This project uses Streamlit for the user interface, LangChain for retrieval-augmented generation, HuggingFace for LLMs, and FAISS for fast vector search.

---
![WhatsApp Image 2025-07-05 at 22 29 25_675eae33](https://github.com/user-attachments/assets/cda84bde-f42b-4c40-aa3a-969198c4ada2)
![WhatsApp Image 2025-07-05 at 22 40 22_4774f594](https://github.com/user-attachments/assets/cda07542-5146-4410-b186-b631d614568d)
![WhatsApp Image 2025-07-05 at 22 41 51_f2e7e683](https://github.com/user-attachments/assets/a796f8d5-dca8-4de9-904a-8229e2ea0384)
![WhatsApp Image 2025-07-05 at 22 46 28_791f1800](https://github.com/user-attachments/assets/5a356d0a-9813-4636-805d-82125ce4dedf)


## 🚀 Features
- **Conversational UI**: Chatbot interface built with Streamlit
- **Medical Knowledge Base**: Uses The Gale Encyclopedia of Medicine (PDF)
- **Semantic Search**: Retrieves relevant context using FAISS vector store and sentence-transformer embeddings
- **LLM-Powered Answers**: Generates detailed, context-aware responses using HuggingFace LLMs (e.g., Mistral-7B-Instruct)
- **Citations**: Each answer includes academic-style references with section summary, page number, and source
- **Customizable Output Length**: Choose between short and medium-form answers
- **Source Transparency**: See exactly where each answer comes from

---

## 📂 Project Structure

```
Medical_chatbot/
├── data/
│   └── The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf
├── vectorstore/
│   └── db_faiss/           # FAISS vector database
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── ...
```

---

## ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Medical_chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your HuggingFace API token**
   - Create a `.env` file in the project root:
     ```
     HF_TOKEN=your_huggingface_token_here
     ```

4. **Prepare the vectorstore**
   - Ensure `The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf` is in the `data/` folder.
   - Run the script to build the vectorstore (if not already built):
     ```bash
     python memory-setup-llm.py
     ```

5. **Run the Streamlit app**
   ```bash
   streamlit run medbot.py
   ```
   - Open your browser to [http://localhost:8501](http://localhost:8501)

---

## 🧑‍⚕️ Usage Guide

- **Ask medical questions**: Type your question in the chat input (e.g., "What are the symptoms of diabetes?")
- **View references**: Each answer includes academic-style references showing the section, page, and source
- **Clear chat**: Use the sidebar button to reset the conversation

---

## 🛠️ Customization

- **Change the LLM**: Edit the model name in `medbot.py` (e.g., use a different HuggingFace model)
- **Adjust retrieval**: Change the number of retrieved documents (`k`) for more or less context
- **Modify prompt template**: Tweak the instructions in `medbot.py` for different answer styles
- **Add more data**: Ingest additional medical PDFs by updating the data loading and vectorstore scripts

---

## 📖 Citation Format

References in answers follow this format:

```
[1] Section summary, in The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf, p. 287.
```

---

## ❓ Troubleshooting

- **App won't start**: Ensure all dependencies are installed and your HuggingFace token is valid
- **No answers or empty responses**: Check that the vectorstore is built and the PDF is in the correct location
- **Model errors**: Make sure the selected HuggingFace model supports text generation and your token has access
- **Slow responses**: Try a smaller model or reduce `max_new_tokens` in the LLM config
- **Formatting issues**: If answers show HTML tags (like `<br>`), ensure the cleaning function replaces them with newlines

---


**Disclaimer:** This chatbot is for informational purposes only and does not provide medical advice. Always consult a healthcare professional for medical concerns. 
