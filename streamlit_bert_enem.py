import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import warnings
import logging

logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')


class BERTTimbauRegression(nn.Module):
    def __init__(self):
        super(BERTTimbauRegression, self).__init__()
        # Load the pre-trained BERT model
        self.bert = BertModel.from_pretrained('neuralmind/bert-base-portuguese-cased')

        # Dropout layer
        self.dropout = nn.Dropout(0.3)

        # A Linear layer to get 5 continuous values
        self.regressor = nn.Linear(768, 5)

    def forward(self, input_ids, attention_mask):
        # Get the output from BERT model
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        pooled_output = outputs.pooler_output

        # Apply dropout
        pooled_output = self.dropout(pooled_output)

        # Pass through the regressor
        return self.regressor(pooled_output)

def preprocess(theme, essay, tokenizer=tokenizer, max_len=512):
    """
    Preprocess a single essay and its theme for inference.

    Args:
    - theme (str): The theme of the essay.
    - essay (str): The essay text.
    - tokenizer: The tokenizer used during training.
    - max_len (int): The maximum length of the tokenized output.

    Returns:
    - A dictionary with keys 'input_ids' and 'attention_mask'.
    """
    # Encoding the theme and essay
    encoding = tokenizer.encode_plus(
        theme,
        essay,
        add_special_tokens=True,
        max_length=max_len,
        truncation=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_overflowing_tokens=False,
        return_tensors='pt',
    )

    return {
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten()
    }

def perform_inference(model, theme, essay,device='cpu'):
    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')

    # Tokenize the input text and prepare input tensors
    inputs = tokenizer.encode_plus(
        theme,
        essay,
        add_special_tokens=True,
        max_length=512,
        truncation=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_overflowing_tokens=False,
        return_tensors='pt',
    )

    # Move tensors to the correct device
    ids = inputs['input_ids'].to(device)
    mask = inputs['attention_mask'].to(device)

    # Set model to evaluation mode
    model.eval()

    # Perform inference
    with torch.no_grad():
        outputs = model(ids, mask)

    # Convert output to numpy array or other suitable format
    outputs = outputs.cpu().detach().numpy()

    return outputs
# Function to load your model (adjust as per your model's requirements)
#@st.cache_data(allow_output_mutation=True)
def load_model():
    # Load the model
    # e.g., model = joblib.load('your_model.pkl') or any other model file
    model = torch.load('bert_enem_final.pt', map_location=torch.device('cpu'))
    return model

model = load_model()

def round_to_nearest_grade(value):
    grades = [0, 40, 80, 120, 160, 200]
    closest_grade = min(grades, key=lambda x: abs(x - value))
    return closest_grade

# Streamlit application layout
st.set_page_config(layout="wide", page_title="Essay Grading Application")

# Title and Intro
st.title("Correção Automática de Redações do ENEM")
st.write("\n")
st.subheader("Insira a sua redação e o tema/proposta para obter a nota.")
st.write("\n")
# Sidebar for application instructions or additional info

# Layout: Use columns to create a more organized layout
col1, col2 = st.columns((1, 1))

# Input fields for the essay and theme
with col1:
    st.write("\n")
    theme = st.text_input("Tema da redação", placeholder="Digite o tema da redação aqui...")
    st.write("\n")
    essay = st.text_area("Texto dissertativo", placeholder="Digite sua redação aqui...", height=300)

# Placeholder for displaying results
with col2:
    result_placeholder = st.empty()

# Button to grade the essay
if st.button('Avaliar redação'):
    if not essay or not theme:
        st.warning("Por favor, insira o tema e a redação.")
    else:
        # Preprocess the essay
        input_text = preprocess(theme, essay)
        
        # Make prediction
        predictions = perform_inference(model, theme, essay, device)[0] * 200  # Assume this returns a NumPy array or a list of scores
        predictions = [round_to_nearest_grade(score) for score in predictions]
        # Display results in the second column
        with result_placeholder.container():
            st.subheader("Resultado da correção automática...")
            for i, score in enumerate(predictions, start=0):
                st.write(f"Competência {i}: {score:.2f}")

            #st.write(f"Total Score: {total_score:.2f}")

            # Create a DataFrame for the scores
            

            # Visualizing the scores as a bar chart
            

# Run the following command in your terminal to start the Streamlit app
# streamlit run your_script_name.py