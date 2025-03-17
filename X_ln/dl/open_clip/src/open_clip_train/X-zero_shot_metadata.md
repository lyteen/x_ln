Lan: `py` From`dl/open_clip/src/open_clip_train\zero_shot_metadata.py`

好的，理解。现在重新开始，目标是提供更智能，组织更清晰的代码。

**1.  增强型提示模板生成器 (Enhanced Prompt Template Generator)**

```python
import random

class PromptTemplateGenerator:
    """
    生成多样化提示模板，用于CLIP模型进行图像分类或生成任务。
    """

    def __init__(self, templates, classnames, n_templates_per_class=5):
        self.templates = templates  # 提示模板列表
        self.classnames = classnames  # 类别名称列表
        self.n_templates_per_class = n_templates_per_class # 每个类别生成的模板数量

    def generate_prompts(self):
        """
        为每个类别生成多个带提示的文本描述。
        """
        prompts = []
        for classname in self.classnames:
            # 随机选择提示模板
            selected_templates = random.sample(self.templates, self.n_templates_per_class)
            for template in selected_templates:
                prompts.append(template(classname))  # 将类别名插入模板
        return prompts

    def generate_prompts_by_class(self, classname):
        """
        为特定的类别生成多个带提示的文本描述。
        """
        prompts = []
        # 随机选择提示模板
        selected_templates = random.sample(self.templates, self.n_templates_per_class)
        for template in selected_templates:
            prompts.append(template(classname))  # 将类别名插入模板
        return prompts

# Demo
if __name__ == '__main__':
    # 假设的提示模板和类别名
    my_templates = [
        lambda c: f'A photo of a {c}.',
        lambda c: f'A painting of the {c}.',
        lambda c: f'A blurry image of a {c}.'
    ]
    my_classnames = ["dog", "cat", "bird"]

    generator = PromptTemplateGenerator(my_templates, my_classnames, n_templates_per_class=2)
    generated_prompts = generator.generate_prompts()

    for prompt in generated_prompts:
        print(prompt)

    #为特定类别生成prompt
    dog_prompts = generator.generate_prompts_by_class("dog")
    for prompt in dog_prompts:
        print(prompt)

```

**描述:** 这个类旨在更灵活地生成各种提示，允许控制每个类别的提示数量，并提供按类别生成提示的功能。

*   **`__init__`**: 初始化函数，接受提示模板列表, 类别名称列表和每个类别生成的模板数量。
*   **`generate_prompts`**: 为所有类别生成提示。 它随机选择提示模板，并将每个类别名称插入到选定的模板中。
*   **`generate_prompts_by_class`**: 为给定的类别生成提示。

---

**2. CLIP文本特征提取器 (CLIP Text Feature Extractor)**

```python
import torch
from transformers import CLIPTokenizer, CLIPModel

class CLIPTextFeatureExtractor:
    """
    使用CLIP模型提取文本特征。
    """
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.device = device

    def extract_features(self, prompts):
        """
        从给定的文本提示中提取CLIP特征。
        """
        inputs = self.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        outputs = self.model.get_text_features(**inputs)  # shape [num_prompts, feature_dim]
        return outputs

# Demo
if __name__ == '__main__':
    # 示例用法
    prompts = ["A photo of a dog.", "A painting of a cat."]
    extractor = CLIPTextFeatureExtractor()
    features = extractor.extract_features(prompts)

    print(f"CLIP文本特征的形状: {features.shape}") # 应为 [2, 512] 对于 base 模型

```

**描述:** 这个类封装了CLIP模型，用于提取文本特征。

*   **`__init__`**: 初始化函数，加载预训练的CLIP模型和tokenizer。
*   **`extract_features`**: 接受文本提示列表，并返回CLIP文本特征。  使用tokenizer将提示转换为模型可以理解的输入，然后通过CLIP模型获取文本特征。

---
**3. 使用增强型提示和CLIP进行零样本分类 (Zero-Shot Classification)**

```python
import torch.nn.functional as F

def zero_shot_classify(image_features, text_features):
    """
    使用 CLIP 特征执行零样本图像分类。

    Args:
        image_features (torch.Tensor):  CLIP图像特征. Shape: [batch_size, feature_dim]
        text_features (torch.Tensor):   CLIP文本特征. Shape: [num_classes, feature_dim]

    Returns:
        torch.Tensor: 每个图像的类概率。Shape: [batch_size, num_classes]
    """
    # 归一化特征
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # 计算相似度
    similarity = image_features @ text_features.T
    probabilities = F.softmax(similarity, dim=-1) # 得到概率

    return probabilities

# Demo
if __name__ == '__main__':
    # 假设的图像和文本特征
    image_features = torch.randn(10, 512)  # 10张图像的特征
    text_features = torch.randn(3, 512)  # 3个类别的特征

    probabilities = zero_shot_classify(image_features, text_features)
    print(f"分类概率的形状: {probabilities.shape}") # 应为 [10, 3]

    predicted_classes = torch.argmax(probabilities, dim=1)
    print(f"预测的类别: {predicted_classes}")

```

**描述:** 这个函数使用 CLIP 特征执行零样本图像分类。

*   **`zero_shot_classify`**: 接受图像特征和文本特征，计算它们之间的相似度，并使用softmax函数将相似度转换为概率。

---

**整体 Demo 示例**

```python
import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

# 步骤 1: 定义类名和提示模板
classnames = ["dog", "cat", "bird"]
templates = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a painting of a {c}.",
    lambda c: f"an image of the {c}."
]

# 步骤 2: 生成提示
prompt_generator = PromptTemplateGenerator(templates, classnames, n_templates_per_class=2)
prompts = prompt_generator.generate_prompts()

# 步骤 3: 提取文本特征
text_extractor = CLIPTextFeatureExtractor()
text_features = text_extractor.extract_features(prompts)
text_features = text_features.reshape(len(classnames), -1, text_features.shape[-1]).mean(dim=1) # 平均每个类的多个提示的特征

# 步骤 4: 加载图像并提取图像特征
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(text_extractor.device)

# 从文件加载图像
image = Image.open("images/cat.jpg")  # 替换为你自己的图像路径

# 准备图像
inputs = processor(text=None, images=image, return_tensors="pt").to(text_extractor.device)
image_features = model.get_image_features(**inputs)

# 步骤 5: 进行零样本分类
probabilities = zero_shot_classify(image_features, text_features)

# 步骤 6: 打印预测结果
predicted_class = torch.argmax(probabilities, dim=1).item()
print(f"预测的类别: {classnames[predicted_class]}")
```

**描述:** 此示例将所有代码段组合在一起以执行零样本图像分类。 它生成提示，提取文本和图像特征，计算相似度，并预测类别。

这个结构允许更大的灵活性和可维护性。例如，可以轻松地更换不同的 CLIP 模型，添加额外的提示模板，或修改图像预处理步骤。
