from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from PIL import Image
import tempfile
from overeasy.types import MultimodalLLM
from typing import Literal
import importlib
import torch

from overeasy.types.base import OCRModel

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
# use cuda device

def setup_autogptq():
    import subprocess
    import torch

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting installation.")
        return

    # Get CUDA version from PyTorch
    cuda_version = torch.version.cuda
    if cuda_version:
        if cuda_version.startswith("11.8"):
            subprocess.run(["pip", "install", "optimum"], check=True)
            subprocess.run([
                "pip", "install", "auto-gptq", "--no-build-isolation",
                "--extra-index-url", "https://huggingface.github.io/autogptq-index/whl/cu118/"
            ], check=True)
        elif cuda_version.startswith("12.1"):
            subprocess.run(["pip", "install", "optimum"], check=True)
            subprocess.run([
                "pip", "install", "auto-gptq", "--no-build-isolation"
            ], check=True)
        else:
            print(f"Unsupported CUDA version: {cuda_version}")
    else:
        print("CUDA version could not be determined.")
        

model_TYPE = Literal["base", "int4", "fp16", "bf16"]
def load_model(model_type: model_TYPE):
    if model_type == "base":
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()
    elif model_type == "fp16":
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True, fp16=True).eval()
    elif model_type == "bf16":
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True, bf16=True).eval()
    elif model_type == "int4":
        def is_autogptq_installed():
            package_name = 'auto_gptq'
            spec = importlib.util.find_spec(package_name)
            return spec is not None

        if not is_autogptq_installed():
            setup_autogptq()
        
        if not is_autogptq_installed():
            raise Exception("AutoGPTQ is not installed can't use int4 quantization")
        
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen-VL-Chat-Int4",
            device_map="cuda",
            trust_remote_code=True
        ).eval()
    else:
        raise Exception("Model type not supported")
        
    # model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
    
    return model

class QwenVL(MultimodalLLM, OCRModel):
    
    def __init__(self, model_type: model_TYPE = "bf16"):
        if not torch.cuda.is_available():
            raise Exception("CUDA not available. Can't use QwenVL")
        if model_type not in ["base", "int4", "fp16", "bf16"]:
            raise Exception("Model type not supported")
        self.model_type = model_type
 
    def load_resources(self):
        self.model = load_model(self.model_type)

        if self.model_type == "int4":
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat-Int4", trust_remote_code=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
       
    def release_resources(self):
        self.model = None
        self.tokenizer = None

    def prompt_with_image(self, image : Image.Image, query: str) -> str:

        with tempfile.NamedTemporaryFile(suffix=".png") as temp_file:
            image.save(temp_file.name)
            query = self.tokenizer.from_list_format([
                {'image': temp_file.name},
                {'text': query},
            ])
            response, history = self.model.chat(self.tokenizer, query=query, history=None, max_new_tokens=2048)
            return response

    def prompt(self, query: str) -> str:
        query = self.tokenizer.from_list_format([
            {'text': query},
        ])
        response, history = self.model.chat(self.tokenizer, query=query, history=None, max_new_tokens=2048)
        return response
    
    def parse_text(self, image: Image.Image):
        return self.prompt_with_image(image, "Read the text in this image line by line")

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