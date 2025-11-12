# =========================================
# Local Chatbot with Ollama + Gradio
# Zero-cost, fully local AI assistant
# =========================================

# --- STEP 0: Install dependencies ---
# !pip install gradio langchain

print("Building a local chatbot with Gradio + Ollama\n")

from langchain_ollama import OllamaLLM
import gradio as gr

# --- Step 1: Initialize local model ---
print("Connecting to Ollama...")
llm = OllamaLLM(model="mistral")
print("✅ Model loaded: Mistral\n")

# --- Step 2: Define chat function ---
def chat_with_model(prompt, history=[]):
    """
    Process user input and generate response

    Args:
        prompt: User's message
        history: Conversation history (list of tuples)

    Returns:
        Updated history
    """
    if not prompt.strip():
        return history, history

    # Generate response
    response = llm(prompt)

    # Append to history
    history.append((prompt, response))

    return history, history

# --- Step 3: Create Gradio chat UI ---
print("🎨 Building user interface...")

with gr.Blocks(title="Local AI Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🧠 Local Chatbot (Ollama + LangChain)
        
        **100% Free • 100% Local • 100% Private**
        
        This chatbot runs entirely on your computer using Ollama.
        No data is sent to external servers.
        """
    )

    chatbot = gr.Chatbot(
        label="Conversation",
        height=400,
        show_label=True
    )

    with gr.Row():
        msg = gr.Textbox(
            label="Your message",
            placeholder="Type your question here...",
            lines=2,
            scale=4
        )
        submit = gr.Button("Send", variant="primary", scale=1)

    clear = gr.ClearButton([msg, chatbot], value="Clear Conversation")

    # Examples section
    gr.Examples(
        examples=[
            "Explain quantum computing in simple terms",
            "Write a Python function to calculate fibonacci numbers",
            "What are the benefits of meditation?",
            "How does photosynthesis work?"
        ],
        inputs=msg,
        label="Try these examples:"
    )

    # Footer info
    gr.Markdown(
        """
        ---
        💡 **Tips:**
        - Be specific in your questions for better answers
        - Ask follow-up questions to dive deeper
        - Experiment with different phrasings
        
        🔧 **Model:** Mistral (running locally via Ollama)
        """
    )

    # Connect events
    msg.submit(chat_with_model, [msg, chatbot], [chatbot, chatbot])
    submit.click(chat_with_model, [msg, chatbot], [chatbot, chatbot])

# --- Step 4: Launch the interface ---
print("✅ Interface ready!")
print("\n" + "=" * 60)
print("🚀 Launching Gradio interface...")
print("📱 Access the chatbot at: http://127.0.0.1:7860")
print("🛑 Press Ctrl+C to stop the server")
print("=" * 60 + "\n")

demo.launch(
    share=False,      # Set to True to create public link (temporary)
    server_name="127.0.0.1",
    server_port=7860
)