from PIL import Image, ImageDraw
import gradio as gr

from vision_engine import run_task
from formatter import (
    format_detailed_caption,
    format_detect_all_objects,
    format_detect_custom_object,
    format_ground_phrase,
    format_ocr_text,
    format_scene_analysis_report,
    format_ask_about_image,
)


def draw_boxes(image, parsed, key, color="red"):
    output_image = image.copy()
    draw = ImageDraw.Draw(output_image)

    try:
        detections = parsed.get(key, {})
        bboxes = detections.get("bboxes", [])
        labels = detections.get("labels", [])

        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            label = labels[i] if i < len(labels) else "object"

            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1, max(0, y1 - 15)), str(label), fill=color)

    except Exception:
        pass

    return output_image


def format_model_evaluation(caption, counts):
    lines = []
    lines.append("Model Evaluation Report")
    lines.append("=======================")
    lines.append("")
    lines.append("Caption:")
    lines.append(caption)
    lines.append("")

    lines.append("Detected Objects:")
    if counts:
        for obj, count in counts.items():
            lines.append(f"- {count} × {obj}")
    else:
        lines.append("- No objects detected")

    lines.append("")
    lines.append("Observations:")

    if counts:
        lines.append("- The model successfully detected several visible objects.")
    else:
        lines.append("- The model struggled to detect clear objects.")

    if "person" in caption.lower():
        lines.append("- Human activity appears to be recognized correctly.")

    lines.append("- Performance may vary depending on image quality and scene complexity.")

    return "\n".join(lines)


def process_action(image, action, user_text):
    if image is None:
        return "Please upload an image first.", None

    if action == "Detailed Caption":
        image, parsed, error = run_task(image, "<DETAILED_CAPTION>")
        if error:
            return error, image
        summary = format_detailed_caption(parsed)
        return summary, image

    if action == "Detect All Objects":
        image, parsed, error = run_task(image, "<OD>")
        if error:
            return error, image
        output_image = draw_boxes(image, parsed, "<OD>", "red")
        summary = format_detect_all_objects(parsed)
        return summary, output_image

    if action == "Detect Custom Object":
        if not user_text.strip():
            return "Please type an object name like: person, car, dog.", image

        image, parsed, error = run_task(image, "<OPEN_VOCABULARY_DETECTION>", user_text)
        if error:
            return error, image

        output_image = draw_boxes(image, parsed, "<OPEN_VOCABULARY_DETECTION>", "lime")
        summary = format_detect_custom_object(parsed, user_text)
        return summary, output_image

    if action == "Ground Phrase":
        if not user_text.strip():
            return "Please type a phrase like: woman, child, laptop.", image

        image, parsed, error = run_task(image, "<CAPTION_TO_PHRASE_GROUNDING>", user_text)
        if error:
            return error, image

        output_image = draw_boxes(image, parsed, "<CAPTION_TO_PHRASE_GROUNDING>", "cyan")
        summary = format_ground_phrase(parsed, user_text)
        return summary, output_image

    if action == "OCR Text":
        image, parsed, error = run_task(image, "<OCR>")
        if error:
            return error, image

        summary = format_ocr_text(parsed)
        return summary, image

    if action == "Scene Analysis Report":
        image, caption_parsed, error = run_task(image, "<DETAILED_CAPTION>")
        if error:
            return error, image

        _, detection_parsed, error = run_task(image, "<OD>")
        if error:
            return error, image

        summary = format_scene_analysis_report(caption_parsed, detection_parsed)
        return summary, image

    if action == "Ask About Image":
        if not user_text.strip():
            return "Please type a question about the image.", image

        question = user_text.strip()
        question_lower = question.lower()

        image, caption_parsed, error = run_task(image, "<MORE_DETAILED_CAPTION>")
        if error:
            return error, image

        caption = caption_parsed.get("<MORE_DETAILED_CAPTION>", "No image understanding available.")

        if "how many" in question_lower and ("people" in question_lower or "persons" in question_lower):
            _, detection_parsed, error = run_task(image, "<OD>")
            if error:
                return error, image

            labels = detection_parsed.get("<OD>", {}).get("labels", [])
            people_count = sum(
                1 for label in labels
                if "person" in label.lower()
            )

            if people_count > 0:
                answer = f"I detected about {people_count} person object(s) in the image."
            else:
                answer = "I could not confidently detect any people in the image."

            return format_ask_about_image(question, answer), image

        if question_lower.startswith("is there ") or question_lower.startswith("are there "):
            if question_lower.startswith("is there "):
                target = question_lower.replace("is there ", "").strip().rstrip("?")
            else:
                target = question_lower.replace("are there ", "").strip().rstrip("?")

            _, detection_parsed, error = run_task(image, "<OPEN_VOCABULARY_DETECTION>", target)
            if error:
                return error, image

            found = detection_parsed.get("<OPEN_VOCABULARY_DETECTION>", {}).get("bboxes", [])

            if found:
                answer = f"Yes, I found {len(found)} match(es) for '{target}'."
            else:
                answer = f"No, I could not find a clear match for '{target}'."

            return format_ask_about_image(question, answer), image

        return format_ask_about_image(question, caption), image

    if action == "Model Evaluation":
        image, caption_parsed, error = run_task(image, "<DETAILED_CAPTION>")
        if error:
            return error, image

        caption = caption_parsed.get("<DETAILED_CAPTION>", "No caption generated.")

        _, detection_parsed, error = run_task(image, "<OD>")
        if error:
            return error, image

        labels = detection_parsed.get("<OD>", {}).get("labels", [])

        counts = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1

        summary = format_model_evaluation(caption, counts)
        return summary, image

    return "Unknown action.", image


custom_css = """
body, .gradio-container {
    background: linear-gradient(135deg, #0b1220 0%, #0f172a 50%, #0a1020 100%) !important;
    color: #f8fafc !important;
    font-family: "Segoe UI", Arial, sans-serif !important;
}

.gradio-container {
    max-width: 1320px !important;
    margin: 0 auto !important;
    padding: 22px 18px 30px 18px !important;
}

h1, h2, h3, p, li, label {
    color: #f8fafc !important;
}

h1 {
    text-align: center !important;
    font-size: 2.65rem !important;
    font-weight: 800 !important;
    margin: 0 !important;
}

.gr-markdown {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    overflow: visible !important;
}

.topbar {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 18px 22px;
    margin-bottom: 14px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.22);
}

.info-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 16px 18px;
    margin-bottom: 14px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.18);
}

.main-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 16px;
    box-shadow: 0 10px 28px rgba(0,0,0,0.20);
}

.inner-card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 14px;
}

.gr-row {
    gap: 16px !important;
}

.gr-column {
    gap: 14px !important;
}

.gr-image, .gr-textbox, .gr-dropdown {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    background: rgba(15,23,42,0.72) !important;
}

.gr-image {
    min-height: 280px !important;
}

textarea, input, select {
    background: rgba(15, 23, 42, 0.92) !important;
    color: #f8fafc !important;
    border: 1px solid rgba(148,163,184,0.28) !important;
    border-radius: 12px !important;
    font-size: 0.98rem !important;
}

textarea::placeholder, input::placeholder {
    color: #94a3b8 !important;
}

.gr-textbox textarea {
    line-height: 1.45 !important;
    padding: 12px !important;
}

.single-line textarea {
    min-height: 52px !important;
    max-height: 52px !important;
    resize: none !important;
    overflow: hidden !important;
}

.output-box textarea {
    min-height: 265px !important;
}

button {
    background: linear-gradient(135deg, #4f46e5, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 700 !important;
    font-size: 1.03rem !important;
    min-height: 48px !important;
    box-shadow: 0 10px 24px rgba(37,99,235,0.26) !important;
    transition: 0.2s ease !important;
}

button:hover {
    filter: brightness(1.07) !important;
    transform: translateY(-1px) !important;
}

footer {
    display: none !important;
}
"""

with gr.Blocks(title="VisionLens AI Pro", css=custom_css) as demo:
    with gr.Column(elem_classes=["topbar"]):
        gr.Markdown("# VisionLens AI Pro")
        gr.Markdown("Interactive Vision-Language AI system")

    with gr.Column(elem_classes=["info-card"]):
        gr.Markdown(
            """
### System Overview
This project uses the **Florence-2 vision-language model** to analyze images through multiple AI tasks.

Supported capabilities:
- Detailed image captioning
- Object detection
- Custom object search
- Phrase grounding
- OCR text extraction
- Scene analysis
- Question answering about the image
- Simple model evaluation

### Bounding Box Legend
- **Red** → Detect All Objects  
- **Lime** → Detect Custom Object  
- **Cyan** → Ground Phrase
"""
        )

    with gr.Row():
        with gr.Column(scale=1, min_width=470, elem_classes=["main-card"]):
            image_input = gr.Image(type="pil", label="Upload Image", height=280)

            with gr.Column(elem_classes=["inner-card"]):
                action = gr.Dropdown(
                    choices=[
                        "Detailed Caption",
                        "Detect All Objects",
                        "Detect Custom Object",
                        "Ground Phrase",
                        "OCR Text",
                        "Scene Analysis Report",
                        "Ask About Image",
                        "Model Evaluation"
                    ],
                    value="Detailed Caption",
                    label="Choose AI Task"
                )

                user_text = gr.Textbox(
                    label="Optional Input",
                    lines=1,
                    max_lines=1,
                    elem_classes=["single-line"],
                    placeholder="Examples: person, car, dog, laptop, woman, child... or ask a question about the image"
                )

            with gr.Column(elem_classes=["inner-card"]):
                gr.Markdown(
                    """
Example object queries:
person, car, dog, laptop, woman, child

Example questions:
How many people are in the image?
Is there a car?
What is happening in this image?
"""
                )

            run_button = gr.Button("Run AI")

        with gr.Column(scale=1, min_width=470, elem_classes=["main-card"]):
            image_output = gr.Image(type="pil", label="Output Image", height=280)
            output_box = gr.Textbox(
                label="AI Output",
                lines=16,
                elem_classes=["output-box"]
            )

    run_button.click(
        fn=process_action,
        inputs=[image_input, action, user_text],
        outputs=[output_box, image_output]
    )

demo.launch()
