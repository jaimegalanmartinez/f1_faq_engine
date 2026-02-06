"""Gradio app entrypoint for the F1 Knowledge Engine."""

import os
import sys
import atexit
import asyncio
import concurrent.futures
import warnings
from dotenv import load_dotenv

import gradio as gr
from rag_pipeline import F1RAGPipeline, RAGConfig

QUERY_TIMEOUT_SECONDS = 30
APP_CSS_PATH = os.path.join(os.path.dirname(__file__), "app.css")
with open(APP_CSS_PATH, "r", encoding="utf-8") as handle:
    CUSTOM_CSS = handle.read()

# Suppress noisy asyncio unclosed event loop warnings from Gradio shutdown.
warnings.filterwarnings(
    "ignore",
    message=r"unclosed event loop.*",
    category=ResourceWarning,
)

env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)

# --- CONFIGURATION ---
WEAVIATE_URL = os.environ.get("WEAVIATE_URL")
WEAVIATE_API_KEY = os.environ.get("WEAVIATE_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")

if not WEAVIATE_URL or not WEAVIATE_API_KEY:
    print("❌ CRITICAL ERROR: Missing Weaviate Credentials in .env file.")
    print("Please create a .env file with WEAVIATE_URL and WEAVIATE_API_KEY.")
    sys.exit(1)  # Stop the script from running


print("🔄 Initializing F1 RAG System...")
rag_config = RAGConfig(
    weaviate_url=WEAVIATE_URL,
    weaviate_api_key=WEAVIATE_API_KEY,
    hf_token=HF_TOKEN)
try:
    rag_pipeline = F1RAGPipeline(rag_config)
    print("✅ System Ready\n")
except Exception as exc:
    print(f"❌ Initialization failed: {exc}")
    sys.exit(1)


def cleanup():
    try:
        rag_pipeline.close()
    except Exception:
        pass
    try:
        loop = asyncio.get_event_loop_policy().get_event_loop()
    except RuntimeError:
        loop = None
    if loop and not loop.is_closed():
        try:
            tasks = asyncio.all_tasks(loop)
            for task in tasks:
                task.cancel()
            if tasks:
                loop.run_until_complete(
                    asyncio.gather(*tasks, return_exceptions=True)
                )
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        loop.close()
    try:
        asyncio.set_event_loop(None)
    except Exception:
        pass


atexit.register(cleanup)


def create_ui():
    """Create Gradio UI interface"""

    with gr.Blocks(title="F1 Knowledge Engine", fill_height=True, fill_width=True) as demo:

        gr.Markdown(
            """
            # Formula 1 Race History Assistant 🏁🚥🏆🏎️💨

            Ask specific questions about race results, winners, and drivers (1950-2024).
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="F1 Historian AI", scale=1, elem_id="chatbot")
                with gr.Row():
                    msg = gr.Textbox(
                        label="Question",
                        placeholder="Ask about specific races or winners...",
                        scale=4)
                    with gr.Column():
                        submit = gr.Button("Send", scale=1, variant="primary")
                        clear = gr.Button("Clear chat", variant="secondary")

            with gr.Column(scale=1):
                gr.Markdown("### 💡 Try asking:")
                gr.Examples(
                    examples=[
                        "Who won the Monaco Grand Prix in 2024?",
                        "Did Lewis Hamilton win at Silverstone in 2008?",
                        "Which team won the Japanese Grand Prix in 2005?",
                        "Who was the first place finisher in the 1950 British GP?",
                        "How many points did Verstappen score at Spa 2023?"
                    ],
                    inputs=msg
                )

                gr.Markdown(
                    """
                    ### 📊 Features
                    - RAG Hybrid search (vector + keyword)
                    - Structured filters (year, race, driver, constructor, circuit)

                    ### ⚙️ System
                    - Embedding: BGE-M3
                    - Vector DB: Weaviate
                    - LLM: HF Router

                    ### Limitations
                    - Driver and constructor standings not included yet
                    - Race results (1950 - 2024)

                    **Built by:** Jaime Galan Martinez<br>
                    **LinkedIn:** https://www.linkedin.com/in/jaimegalanmartinez/<br>
                    **GitHub:** https://github.com/jaimegalanmartinez/f1_faq_engine
                    """
                )

        def respond(message, chat_history):
            # 1. Append the user's message (as a dictionary)
            chat_history.append({"role": "user", "content": message})

            # 2. Placeholder assistant message
            chat_history.append({"role": "assistant", "content": "Thinking..."})
            yield "", chat_history

            # 3. Run RAG query with timeout
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(rag_pipeline.query, message, chat_history)
                    bot_message = future.result(timeout=QUERY_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                bot_message = "⏱️ Timed out. Please try again."
            except Exception:
                bot_message = "⚠️ Something went wrong. Please try again."

            # 4. Replace placeholder with the real response
            chat_history[-1] = {"role": "assistant", "content": bot_message}
            yield "", chat_history

        def clear_chat():
            """Clear the chat history."""
            return []

        submit.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
        msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
        clear.click(clear_chat, outputs=[chatbot])

    return demo


def launch_app():
    app = create_ui()
    try:
        app.launch(
            theme=gr.themes.Soft(),
            css=CUSTOM_CSS,
            server_name="0.0.0.0",
            server_port=7860)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            app.close()
        except Exception:
            pass


if __name__ == "__main__":
    launch_app()
