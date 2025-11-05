import gradio as gr


def create_app(rag_fn) -> gr.blocks.Blocks:
    """Create the Gradio ChatInterface wired to the provided RAG function.

    Design: UI assembly is isolated so we can evolve the interface (inputs,
    outputs, custom components) without touching core initialization logic.
    """
    return gr.ChatInterface(
        fn=rag_fn,
        title="RAG Chatbot",
        description="Ask me anything about the loaded documents!",
        type="messages"
    )
