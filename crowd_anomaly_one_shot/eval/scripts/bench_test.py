import cv2, torch, time
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

model_id = 'HuggingFaceTB/SmolVLM2-2.2B-Instruct'
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    _attn_implementation='sdpa'
).to('cuda')
model.eval()

# Load 1 frame from Arson001 at frame 1000 where fire is burning
cap = cv2.VideoCapture('eval/data/ucf_crime/Arson/Arson001_x264.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 1000)
ret, f = cap.read()
cap.release()
img = Image.fromarray(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (224, 224)))

messages = [{
    'role': 'user',
    'content': [{'type': 'image'}, {'type': 'text', 'text': 'Classify this surveillance scene: Normal, Violence, Fighting, Arson, Car Accident, Explosion, Shooting. Answer with ONLY the category.'}]
}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[img], return_tensors='pt').to('cuda')

t0 = time.time()
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
lat = (time.time() - t0) * 1000
txt = processor.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
print(f'SmolVLM2 Result: \"{txt}\" in {lat:.1f} ms')

