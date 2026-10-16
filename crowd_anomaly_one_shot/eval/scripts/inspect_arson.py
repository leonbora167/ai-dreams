import cv2, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained('HuggingFaceTB/SmolVLM2-2.2B-Instruct')
model = AutoModelForImageTextToText.from_pretrained(
    'HuggingFaceTB/SmolVLM2-2.2B-Instruct',
    torch_dtype=torch.float16,
    _attn_implementation='sdpa'
).to('cuda')
model.eval()

cap = cv2.VideoCapture('eval/data/ucf_crime/Arson/Arson001_x264.mp4')
for f_idx in [50, 150, 250, 500, 1500, 2500, 3500]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ret, frame = cap.read()
    if not ret: continue
    img = Image.fromarray(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (224, 224)))
    messages = [{
        'role': 'user',
        'content': [{'type': 'image'}, {'type': 'text', 'text': 'Describe what is shown in this surveillance frame.'}]
    }]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[img], return_tensors='pt').to('cuda')
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=25, do_sample=False)
    resp = processor.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    print(f'Frame {f_idx}: \"{resp}\"')
cap.release()

