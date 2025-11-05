import gradio as gr


def create_app(rag_fn) -> gr.blocks.Blocks:
    return gr.ChatInterface(
        fn=rag_fn,
        title="RAG Chatbot",
        description="Ask me anything about the loaded documents!",
    )


