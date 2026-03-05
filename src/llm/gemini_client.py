import google.generativeai as genai
from io import BytesIO
import base64
from PIL import Image

class GeminiClient:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
    
    def encode_image(self, fig):
        buf = BytesIO()
        
        if hasattr(fig, 'savefig'):
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            img = Image.open(buf)
        elif hasattr(fig, 'to_image'):
            img_bytes = fig.to_image(format='png')
            buf = BytesIO(img_bytes)
            img = Image.open(buf)
        else:
            raise ValueError(f"Unsupported figure type: {type(fig)}")
        
        return img
    
    def interpret(self, prompt, image=None):
        if image:
            response = self.model.generate_content([prompt, image])
        else:
            response = self.model.generate_content(prompt)
        return response.text

def interpret_visualization(api_key, prompt, figure=None):
    client = GeminiClient(api_key)
    if figure:
        img = client.encode_image(figure)
        return client.interpret(prompt, img)
    return client.interpret(prompt)
