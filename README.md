# VisionLens AI

This project is an interactive vision-language system that analyzes images using the Florence-2 multimodal model.

The user can upload an image and run different tasks such as captioning, object detection, phrase grounding, OCR, and asking questions about the image.

## Tasks Supported
- Detailed Caption
- Detect All Objects
- Detect Custom Object
- Phrase Grounding
- OCR
- Scene Analysis
- Ask About Image

## Architecture
The system is divided into three main parts:

app.py → user interface (Gradio)

vision_engine.py → loads the model and runs inference

formatter.py → formats the output results

## Model
Florence-2 Vision-Language Transformer

## Technologies
Python  
PyTorch  
HuggingFace Transformers  
Gradio
