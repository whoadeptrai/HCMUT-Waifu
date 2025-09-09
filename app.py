# app.py
import gradio as gr
from inference import generate_text

def chat_with_model(prompt):
    if not prompt:
        return ""
    # bạn có thể điều chỉnh max_new_tokens, temperature, top_k ở đây
    return generate_text(prompt, max_new_tokens=200)

demo = gr.Interface(
    fn=chat_with_model,
    inputs=gr.Textbox(lines=2, placeholder="Gõ câu hỏi của bạn..."),
    outputs=gr.Textbox(label="AI trả lời"),
    title="Mini GPT Demo",
    description="Giao diện chat đơn giản với model của bạn"
)

if __name__ == "__main__":
    # share=True tạo 1 link public phù hợp để chạy trên Colab
    demo.launch(share=True)
