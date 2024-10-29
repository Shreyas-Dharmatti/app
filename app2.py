import streamlit as st
pip install torch
#pip install 
st.title("Text Prediction Model")
option = st.selectbox("Choose an option:", ["Sentence Generation", "Word Generation"])

# Sentence Generation Section
if option == "Sentence Generation":
    st.subheader("Sentence Generation")
    import torch
    import torch.nn as nn
    import re
    import requests

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Load text data
    url = "https://cs.stanford.edu/people/karpathy/char-rnn/warpeace_input.txt"
    response = requests.get(url)
    text = response.text

    # Step 1: Mark paragraph breaks with a unique token `<PARA>`
    text = re.sub(r'\n\s*\n', ' <PARA> ', text)  # Replace paragraph breaks with `<PARA>`
    text = "\n".join(line + " " for line in text.splitlines())  # Add space at the end of each line
    text = re.sub('[^a-zA-Z0-9 \.<>]', '', text)  # Keep only alphanumeric, space, period, and tokens
    text = text.lower().strip()

    # Step 2: Split into words, treating full stops and paragraph tokens as separate words
    words = re.findall(r'\b\w+\b|[.]|<PARA>', text)

    # Step 3: Add padding token `<>` and define the vocabulary
    vocab = sorted(set(words + ["<>"]))  # Add `<>` to the vocabulary
    stoi = {s: i for i, s in enumerate(vocab)}  # Map each token to a unique index
    itos = {i: s for s, i in stoi.items()}      # Reverse map from index to token

    # Set up Streamlit interface for user input
    st.title("Next Sentence Prediction Model")
    st.write("Configure the model settings:")

    embedding_dim = st.selectbox("Embedding Dimension", [32,64])
    block_size = st.selectbox("Block Size (Context Length)", [5,10])
    activation_function = st.selectbox("Activation Function", ["ReLU", "Tanh"])
    # Text input for user to provide initial words, limiting to a maximum of 5 words
    input_text = st.text_input(f"Enter up to {block_size} words as context to generate (optional):", placeholder="This is a sample sentence")
    if len(input_text.split()) > block_size:
        st.warning(f"Please enter no more than {block_size} words.")
        input_text = " ".join(input_text.split())  # Truncate to the first 5 words
    input_words = input_text.split()[:block_size]
    input_words = [''] * (block_size - len(input_words)) + input_words
    num_sen = st.number_input("Number of Sentences to Generate", min_value=1, max_value=100000, step=1, value=10)

    # Map the activation function choice to PyTorch activation functions
    activation_dict = {
        "ReLU": nn.ReLU(),
        "Tanh": nn.Tanh(),
    }
    activation_func = activation_dict[activation_function]

    # Define your model architecture
    class NextWordModel(nn.Module):
        def __init__(self, block_size, vocab_size, emb_dim, hidden_size, activation_func):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb_dim)
            self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
            self.activation = activation_func
            self.lin2 = nn.Linear(hidden_size, vocab_size)

        def forward(self, x):
            x = self.emb(x)
            x = x.view(x.shape[0], -1)
            x = self.activation(self.lin1(x))
            x = self.lin2(x)
            return x

    # Initialize the model with the same architecture
    vocab_size = len(stoi)
    model = NextWordModel(block_size, vocab_size, embedding_dim, 1024,activation_func)  # Ensure these match the original model

    model_filename = f"next_word_model_2_{embedding_dim}_{block_size}_{activation_function.lower()}_500.pth"

    model.load_state_dict(torch.load(model_filename, map_location=torch.device('cpu')))

    def predict_next_sentences(model, stoi, itos, input_words, device, num_sentences=1):
        model.eval()  # Set model to evaluation mode
        import random
        random.seed(42)
        context = [stoi.get(word, random.randint(500, 20000)) for word in input_words[-block_size:]]
        #context = [stoi.get(word, 93) for word in input_words[-block_size:]]
        sentence = input_words.copy()
        sentence_count = 0  # Initialize sentence counter

        while sentence_count < num_sentences:
            input_seq = torch.tensor(context, dtype=torch.long).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_seq)
                _, predicted = torch.max(output, dim=1)
                next_word = itos[predicted.item()]

            # If next word is a full stop, check for consecutive full stops
            if next_word == '.':
                # Avoid appending consecutive full stops by checking the last word added
                if sentence[-1] != '.':
                    sentence.append(next_word)
                    sentence_count += 1  # Count as end of a sentence
            else:
                sentence.append(next_word)

            # Update context for the next prediction
            context = context[1:] + [predicted.item()]
        
        # Join and return the generated sentence
        return ' '.join(sentence)

    # # def predict_next_sentences(model, stoi, itos, input_words, device, block_size, num_sentences=1):
    # #     model.eval()  # Set model to evaluation mode
        
    # #     # Ensure context is padded to match block_size
    # #     context = [stoi.get(word, 0) for word in input_words[-block_size:]]
    # #     if len(context) < block_size:
    # #         context = [0] * (block_size - len(context)) + context  # Pad with zeros
        
    # #     sentence = input_words.copy()
    # #     sentence_count = 0  # Initialize sentence counter
    # #     max_attempts = num_sentences * block_size * 5  # Limit attempts to avoid infinite loops

    # #     attempt = 0
    # #     while sentence_count < num_sentences and attempt < max_attempts:
    # #         attempt += 1
            
    # #         # Convert context to tensor and predict the next word
    # #         input_seq = torch.tensor(context, dtype=torch.long).unsqueeze(0).to(device)
            
    # #         with torch.no_grad():
    # #             output = model(input_seq)
    # #             _, predicted = torch.max(output, dim=1)
    # #             next_word = itos.get(predicted.item(), '<UNK>')  # Use `<UNK>` if prediction is out of vocabulary

    # #         # Append the next word and update sentence count if it's a period
    # #         if next_word == '.':
    # #             if sentence and sentence[-1] != '.':
    # #                 sentence.append(next_word)
    # #                 sentence_count += 1  # Count as end of a sentence
    # #         else:
    # #             sentence.append(next_word)

    # #         # Update context for the next prediction
    # #         context = context[1:] + [stoi.get(next_word, 0)]

    # #     return ' '.join(sentence).strip() if sentence else "No valid text generated."
    
    if st.button("Generate Text"):
        st.write("Generated sentence:", predict_next_sentences(model, stoi, itos, input_words, device, num_sen).strip())

# Word Generation Section
elif option == "Word Generation":
    st.subheader("Word Generation")
    # Code for Word Generation would go here
    # Example: st.write("Word Generation content will be implemented here.")
    import torch
    import torch.nn as nn
    import re
    import requests
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    url = "https://cs.stanford.edu/people/karpathy/char-rnn/warpeace_input.txt"
    response = requests.get(url)
    text = response.text

    # Add a space at the end of each line
    text = "\n".join(line + " " for line in text.splitlines())
    text = re.sub('[^a-zA-Z0-9 \.]', '', text)
    text = text.lower().strip()
    words = re.findall(r'\b\w+\b', text)
    vocab = sorted(set(words))  # stores unique words in the training vocabulary
    print(words[:35])
    print(len(words))

    # Create word-to-index and index-to-word mappings
    stoi = {s: i+1 for i, s in enumerate(vocab)}  # '+1' to reserve index 0 for padding
    stoi['.'] = 0  # Reserve index 0 for padding or end token
    itos = {i: s for s, i in stoi.items()}

    # Set up Streamlit interface for user input
    st.title("Next Word Prediction Model")
    st.write("Configure the model settings:")

    embedding_dim = st.selectbox("Embedding Dimension", [32,64])
    block_size = st.selectbox("Block Size (Context Length)", [5,10,15])
    activation_function = st.selectbox("Activation Function", ["ReLU", "Tanh"])
    # Text input for user to provide initial words, limiting to a maximum of 5 words
    input_text = st.text_input(f"Enter up to {block_size} words as context to generate (optional):", placeholder="This is a sample sentence")
    if len(input_text.split()) > block_size:
        st.warning(f"Please enter no more than {block_size} words.")
        input_text = " ".join(input_text.split())  # Truncate to the first 5 words
    input_words = input_text.split()[:block_size]
    input_words = [''] * (block_size - len(input_words)) + input_words
    num_words = st.number_input("Number of Words to Generate", min_value=1, max_value=100000, step=1, value=10)

    # Map the activation function choice to PyTorch activation functions
    activation_dict = {
        "ReLU": nn.ReLU(),
        "Tanh": nn.Tanh(),
    }
    activation_func = activation_dict[activation_function]

    # Define your model architecture
    class NextWordModel(nn.Module):
        def __init__(self, block_size, vocab_size, emb_dim, hidden_size, activation_func):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb_dim)
            self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
            self.activation = activation_func
            self.lin2 = nn.Linear(hidden_size, vocab_size)

        def forward(self, x):
            x = self.emb(x)
            x = x.view(x.shape[0], -1)
            x = self.activation(self.lin1(x))
            x = self.lin2(x)
            return x

    # Initialize the model with the same architecture
    vocab_size = len(stoi)
    model = NextWordModel(block_size, vocab_size, embedding_dim, 1024,activation_func)  # Ensure these match the original model

    model_filename = f"next_word_model_{embedding_dim}_{block_size}_{activation_function.lower()}_500.pth"

    model.load_state_dict(torch.load(model_filename, map_location=torch.device('cpu')))

    def predict_next_words(model, stoi, itos, input_words, device, num_words=5):

        model.eval()  # Set model to evaluation mode
        context = [stoi.get(word, 0) for word in input_words[-block_size:]]
        sentence = input_words.copy()

        for _ in range(num_words):
            input_seq = torch.tensor(context, dtype=torch.long).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_seq)
                _, predicted = torch.max(output, dim=1)
                next_word = itos[predicted.item()]
                sentence.append(next_word)

                # Update context for the next prediction
                context = context[1:] + [predicted.item()]

        return ' '.join(sentence)
    if st.button("Generate Text"):
        st.write("Generated sentence:", predict_next_words(model, stoi, itos, input_words, device, num_words).strip())

