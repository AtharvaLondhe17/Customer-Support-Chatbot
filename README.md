Overview
This project is a simple customer support chatbot built using GPT-2 and fine-tuned for customer service queries. The chatbot responds to user queries using either predefined responses from a small dataset or generates responses using the fine-tuned GPT-2 model. The app is developed using Streamlit for easy deployment and interaction.

Features
Predefined Responses: The bot provides predefined answers for common customer support questions.
AI-Generated Responses: If a predefined response is unavailable, GPT-2 generates a response.
Dataset Management: Add new queries and responses through the Streamlit sidebar.

Requirements
Python 3.7+
Libraries:
streamlit
transformers
torch
datasets
pandas
difflib
shutil
tempfile
