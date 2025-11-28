import easyocr
from PIL import Image
import tempfile
import os

_reader = None

def get_reader(lang_list=['en'], gpu=False):
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(lang_list, gpu=gpu)
    return _reader

def ocr_image_pil(pil_image, lang_list=['en']):
    reader = get_reader(lang_list=lang_list)

   
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
        pil_image.save(tmp.name)
        result = reader.readtext(tmp.name, detail=0)

    try:
        os.remove(tmp.name)
    except:
        pass

    return "\n".join(result)

def ocr_image_path(image_path, lang_list=['en']):
    img = Image.open(image_path)
    return ocr_image_pil(img, lang_list=lang_list)



text = ocr_image_pil(Image.open(r"C:\Users\thouf\Pictures\Screenshots\Screenshot 2025-11-17 151339.png"))
print(text)

