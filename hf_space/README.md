---
title: Xori Conversation Space
emoji: 🤖
colorFrom: purple
colorTo: blue
sdk: gradio
app_file: app.py
---

# Xori Conversation Space

This Space is the remote inference brain for Xori.

Render stays lightweight and forwards conversation requests here. The model weights remain outside Render.

Set `XORI_MODEL_ID` to your Hugging Face model repository. The default is `Qwen/Qwen2.5-0.5B-Instruct`.

The app exposes a named Gradio API endpoint called `generate`. Gradio documents named API endpoints and the `/gradio_api/call/<endpoint>` HTTP flow in its API documentation. 
