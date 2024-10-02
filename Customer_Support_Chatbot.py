import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import pandas as pd
import os
import torch
import difflib
import shutil
import tempfile
import time

# Define the dataset
dataset = [
    {"query": "What are your business hours?", 
     "response": "Our business hours are from 9 AM to 6 PM, Monday through Friday."},
    {"query": "How can I reset my password?", 
     "response": "To reset your password, click on 'Forgot Password' on the login page and follow the instructions."},
    {"query": "Where is your office located?", 
     "response": "Our office is located at 1234 Elm Street, Cityville."},
    {"query": "How can I contact customer support?", 
     "response": "You can contact customer support by emailing support@company.com or calling 1-800-123-4567."},
    {"query": "Do you offer refunds?", 
     "response": "Yes, we offer refunds within 30 days of purchase. Please visit our return policy page for more details."},
    {"query": "How do I track my order?", 
     "response": "You can track your order by logging into your account and clicking on 'Order History'."},
    {"query": "What is your return policy?", 
     "response": "Our return policy allows returns within 30 days of purchase with the original receipt and packaging."},
    {"query": "Do you offer technical support?", 
     "response": "Yes, we offer 24/7 technical support. You can contact us via phone or email for assistance."},
    {"query": "Can I change my delivery address?", 
     "response": "Yes, you can change your delivery address before the item is shipped. Please contact customer support for assistance."},
    {"query": "How long does shipping take?", 
     "response": "Shipping typically takes 3-5 business days for standard delivery."},
    {"query": "What payment methods do you accept?", 
     "response": "We accept credit cards, debit cards, PayPal, and Apple Pay."},
    {"query": "Can I cancel my order?", 
     "response": "You can cancel your order within 24 hours of purchase by contacting our customer service team."},
    {"query": "How do I update my account information?", 
     "response": "To update your account information, log in to your account and go to the 'Account Settings' page."},
    {"query": "Do you offer international shipping?", 
     "response": "Yes, we offer international shipping to select countries. Please check our shipping policy page for details."},
    {"query": "Can I get a discount?", 
     "response": "We offer seasonal discounts and promotions. Please subscribe to our newsletter to stay updated on the latest offers."},
    {"query": "What is your warranty policy?", 
     "response": "Our products come with a 1-year limited warranty. Please visit our warranty page for more information."},
    {"query": "How do I file a complaint?", 
     "response": "To file a complaint, please contact our customer service team at support@company.com. We'll resolve your issue as soon as possible."},
    {"query": "Is my data secure?", 
     "response": "Yes, we take your privacy seriously. All customer data is encrypted and stored securely in compliance with data protection regulations."},
    {"query": "Do you provide installation services?", 
     "response": "Yes, we provide installation services for select products. Please check the product page for availability."},
    {"query": "How do I apply a coupon code?", 
     "response": "You can apply a coupon code during the checkout process by entering it in the 'Discount Code' field."},
]

# Convert the dataset into a Hugging Face Dataset format
hf_dataset = Dataset.from_pandas(pd.DataFrame(dataset))

# Define model directory and load model
@st.cache_resource
def load_model():
    model_dir = "./fine_tuned_gpt2"
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model

# Tokenization function for GPT-2
def tokenize_function(examples, tokenizer):
    queries = examples["query"]
    responses = examples["response"]
    combined = [f"{q} [SEP] {r}" for q, r in zip(queries, responses)]
    return tokenizer(combined, padding="max_length", truncation=True, max_length=128)

def safe_save_model(trainer, output_dir, retries=3, delay=2):
    for attempt in range(retries):
        try:
            # Try to save normally
            trainer.save_model(output_dir)
            st.success("Model saved successfully!")
            return
        except Exception as e:
            st.warning(f"Error saving model (attempt {attempt + 1}): {str(e)}. Retrying...")
            time.sleep(delay)  # Wait before retrying
            
            # Create a temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save the model to the temporary directory
                trainer.save_model(temp_dir)
                
                # Copy files from temp directory to the actual output directory
                for item in os.listdir(temp_dir):
                    s = os.path.join(temp_dir, item)
                    d = os.path.join(output_dir, item)
                    if os.path.isdir(s):
                        shutil.copytree(s, d, dirs_exist_ok=True)
                    else:
                        shutil.copy2(s, d)
            st.success("Model saved successfully!")
            return
            
    st.error("Failed to save the model after multiple attempts.")

# Check if model exists and train if not
# Before training, enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# Training function
def train_model():
    tokenizer, model = load_model()
    
    if not os.path.exists(os.path.join("./fine_tuned_gpt2", "pytorch_model.bin")):
        tokenized_dataset = hf_dataset.map(
            lambda examples: tokenize_function(examples, tokenizer),
            batched=True,
            remove_columns=hf_dataset.column_names
        )

        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        training_args = TrainingArguments(
            output_dir="./fine_tuned_gpt2",
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            save_steps=500,
            save_total_limit=2,
            prediction_loss_only=True,
            logging_dir='./logs'
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        # Clear any pre-existing gradients
        model.zero_grad()
        
        trainer.train()
        safe_save_model(trainer, "./fine_tuned_gpt2")

    return tokenizer, model


# Train or load the model
tokenizer, model = train_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Use half-precision (FP16) if using a GPU
if device.type == 'cuda':
    model.half()

# Create a dictionary of predefined responses
predefined_responses = {item['query']: item['response'] for item in dataset}

def find_best_match(user_query, predefined_queries):
    return difflib.get_close_matches(user_query, predefined_queries, n=1, cutoff=0.6)

def generate_model_response(query):
    input_ids = tokenizer.encode(query + " [SEP]", return_tensors='pt', padding=True, truncation=True, max_length=64).to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_length=128,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.95,
            no_repeat_ngram_size=3,
            do_sample=True,
            early_stopping=True
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.split("[SEP]")[-1].strip()

# Streamlit app
st.title("Customer Support Chatbot")
st.write("Ask me anything related to customer support!")

user_query = st.text_input("Your Query:")

if st.button("Get Response"):
    if user_query:
        # Check for predefined responses
        best_match = find_best_match(user_query, predefined_responses.keys())
        if best_match:
            response = predefined_responses[best_match[0]]
            st.success(f"Chatbot Response: {response}")
        else:
            # Generate response using the model
            response = generate_model_response(user_query)
            st.info(f"Chatbot Response: {response}")
            st.caption("This response was generated by AI and may not be perfectly accurate.")
    else:
        st.error("Please enter a query before clicking the button.")

# Add a section to display and edit the dataset
st.sidebar.title("Dataset Management")
if st.sidebar.checkbox("Show Dataset"):
    st.sidebar.table(dataset)

# Add new entries to the dataset
st.sidebar.subheader("Add New Entry")
new_query = st.sidebar.text_input("New Query:")
new_response = st.sidebar.text_input("New Response:")

if st.sidebar.button("Add Entry"):
    if new_query and new_response:
        dataset.append({"query": new_query, "response": new_response})
        st.sidebar.success("Entry added!")
    else:
        st.sidebar.error("Please fill in both fields.")

