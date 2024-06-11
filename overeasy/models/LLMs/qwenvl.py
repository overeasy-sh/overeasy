from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from PIL import Image
import tempfile
from overeasy.logging import log_time
from overeasy.types import MultimodalLLM

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
# use cuda device
def load_model():
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()
    model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True,)
    return model

class QwenVL(MultimodalLLM):
    def __init__(self):
        self.model = load_model()
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
        
    @log_time
    def prompt_with_image(self, image : Image.Image, query: str) -> str:

        with tempfile.NamedTemporaryFile(suffix=".png") as temp_file:
            image.save(temp_file.name)
            query = tokenizer.from_list_format([
                {'image': temp_file.name},
                {'text': query},
            ])
            response, history = self.model.chat(tokenizer, query=query, history=None)
            return response

    @log_time
    def prompt(self, query: str) -> str:
        query = tokenizer.from_list_format([
            {'text': query},
        ])
        response, history = self.model.chat(tokenizer, query=query, history=None)
        return response

    def draw_bbox_on_latest_picture(self, response, history):
        if response.startswith("<ref>") and response.endswith("</ref>"):
            response = response[5:-6]
            if "<box>" in response and "</box>" in response:
                box = response[response.index("<box>") + 5:response.index("</box>")]
                box = box.split(",")
                box = [int(x) for x in box]
                image = Image.open(history[-1]["image"])
                return image.crop(box)
        return None