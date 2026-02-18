
📁 Project Structure
1-Prototype/
│── appAnalytics.py
│── data/
│     └── file.pdf
│── vector_store/
│── logs/
│     └── chat_logs.csv

1️⃣ Install Ollama

👉 https://ollama.com

Then pull embedding + chat models:

ollama pull llama3
ollama pull nomic-embed-text

2️⃣ Install Python packages
pip install -r requirements.txt

▶️ Run the App
streamlit run appAnalytics.py 
