import google.generativeai as genai
import os
from dotenv import load_dotenv
import gradio as gr
"""
Gradio=5.0.2
Generativeai=0.8.3

"""

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

chat_session = None
context_text = ""

def build_model():
    global chat_session
    generation_config = {
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
    }
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        system_instruction="""Metne göre soruları cevaplayın.
Metinde ilgili bilgi yoksa sadece Bilmiyorum diye cevaplayın.Var ise sadece cevabı yazın."
        """
    )
    chat_session = model.start_chat(history=[])

def set_context(text):
    global context_text
    context_text = text
    build_model()
    return "Context set successfully. You can now ask questions."

def chat(context, question):
    global context_text
    if context != context_text:
        set_context(context)
    
    try:
        response = chat_session.send_message([context, question])
        return response.text
    except Exception as e:
        return f"An error occurred: {str(e)}"

def quit_app():
    iface.close()

with gr.Blocks() as iface:
    gr.Markdown("# Gemini QA System")
    gr.Markdown("metni yapıştır ve sorunu sor.")
    
    with gr.Row():
        context_input = gr.Textbox(lines=5, label="Context", placeholder="Paste your context text here...")
        question_input = gr.Textbox(lines=2, label="Question", placeholder="Ask a question about the context...")
    
    submit_btn = gr.Button("Submit")
    output = gr.Textbox(label="Answer")
    
    submit_btn.click(fn=chat, inputs=[context_input, question_input], outputs=output)

    quit_btn = gr.Button("Quit")
    quit_btn.click(fn=quit_app, inputs=None, outputs=None)
    gr.Examples(
        examples=[
            [" PAROL, her tabletinde 500 mg parasetamol içeren, ağrı kesici ve ateş düşürücü olarak etkieden bir ilaçtır.PAROL 20 ve 30 tablet içeren blister ambalajlardadır.PAROL hafif ve orta şiddetli ağrılar ve ateşin semptomatik  tedavisinde kullanılır.",
             "PAROL nedir ve ne için kullanılır?"]
        ],
        inputs=[context_input, question_input]
    )


iface.launch()