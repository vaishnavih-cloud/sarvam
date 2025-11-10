# ============================================
# 🌍 SarvamAI Multilingual Chatbot (Full Setup)
# ============================================

# Step 1: Import libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr

# Step 2: Load the SarvamAI model
# This is a multilingual model that supports Indian languages
model_name = "sarvamai/sarvam-2b"

print("🔄 Loading SarvamAI model... This may take a few minutes initially.")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Use GPU if available, else CPU
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
print("✅ Model loaded successfully!")

# Step 3: Define the chat generation function
def generate_reply(user_input, history):
    # Combine all previous chat messages
    conversation = ""
    for u, b in history:
        conversation += f"User: {u}\nBot: {b}\n"
    conversation += f"User: {user_input}\nBot:"

    # Tokenize and feed into the model
    inputs = tokenizer(conversation, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode text output
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only bot’s response
    if "Bot:" in text:
        reply = text.split("Bot:")[-1].strip()
    else:
        reply = text.strip()

    # Add to chat history
    history.append((user_input, reply))
    return history, ""

# Step 4: Create a Gradio web interface
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🤖 SarvamAI Multilingual Chatbot  
        Talk in **English, Hindi, Tamil, Telugu, Marathi, Bengali, Kannada, Malayalam, or Gujarati.**
        """
    )
    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox(label="💬 Type your message here...")
    clear = gr.Button("🧹 Clear Chat")

    msg.submit(generate_reply, [msg, chatbot], [chatbot, msg])
    clear.click(lambda: None, None, chatbot, queue=False)

# Step 5: Launch the chatbot
if __name__ == "__main__":
    demo.launch()
