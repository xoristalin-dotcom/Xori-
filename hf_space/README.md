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

The default model is `Abobus2222228/Xoritg`. Set `XORI_MODEL_ID` in Space variables to override it.

The app exposes a named Gradio API endpoint called `generate`.
