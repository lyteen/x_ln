Lan: `py` From`dl/RAG/PIKE-RAG/src\ragflow\api\utils\file_utils.py`

Okay, restarting with improvements and Chinese descriptions. I'll focus on providing more robust and efficient solutions, along with clear explanations and practical examples.  The examples will also incorporate better error handling and resource management.

**1. 改进的PDF缩略图生成 (Improved PDF Thumbnail Generation):**

```python
import base64
import io
import os
import sys
import threading

from PIL import Image
import pdfplumber

from api.constants import IMG_BASE64_PREFIX

LOCK_KEY_pdfplumber = "global_shared_lock_pdfplumber"
if LOCK_KEY_pdfplumber not in sys.modules:
    sys.modules[LOCK_KEY_pdfplumber] = threading.Lock()

def thumbnail_pdf(blob, max_size=64000):
    """从PDF blob生成缩略图."""
    try:
        with sys.modules[LOCK_KEY_pdfplumber]:
            with pdfplumber.open(io.BytesIO(blob)) as pdf:
                page = pdf.pages[0]  # 获取第一页
                im = page.to_image(resolution=32) # 使用 pdfplumber 内置图像转换功能
                image = im.original
                # 如果图像太大，调整分辨率
                buffered = io.BytesIO()
                if len(image.tobytes()) > max_size:  # Check image size instead of byte size
                    # 如果太大，则降低图片质量
                    image.save(buffered, format="png", optimize=True, quality=85)  # Adjust quality for better compression
                else:
                    image.save(buffered, format="png")  # сохранение в буфер

                return buffered.getvalue()

    except Exception as e:
        print(f"PDF 缩略图生成失败: {e}")
        return None

def thumbnail(filename, blob):
    img = thumbnail_img(filename, blob)
    if img is not None:
        return IMG_BASE64_PREFIX + \
            base64.b64encode(img).decode("utf-8")
    else:
        return ''

def thumbnail_img(filename, blob):
    """
    MySQL LongText max length is 65535
    """
    filename = filename.lower()
    if re.match(r".*\.pdf$", filename):
        return thumbnail_pdf(blob)

    elif re.match(r".*\.(jpg|jpeg|png|tif|gif|icon|ico|webp)$", filename):
        image = Image.open(BytesIO(blob))
        image.thumbnail((30, 30))
        buffered = BytesIO()
        image.save(buffered, format="png")
        return buffered.getvalue()

    elif re.match(r".*\.(ppt|pptx)$", filename):
        import aspose.slides as slides
        import aspose.pydrawing as drawing