Lan: `py` From`X_ln\dl/open_clip/src\open_clip\tokenizer.html`

好的，重新开始。

**1. 字节对编码 (Byte Pair Encoding, BPE) 相关的函数**

```python
import gzip
import html
import os
import string
from functools import lru_cache
import ftfy
import regex as re

@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs
```

**描述:** 这些函数主要用于处理字节对编码(BPE)算法，将文本转换为token。
- `default_bpe()`:  返回默认的BPE词汇表文件的路径.  `@lru_cache()`装饰器用于缓存结果，提高性能。
- `bytes_to_unicode()`: 创建一个字节到Unicode字符的映射，用于处理UTF-8编码。
- `get_pairs(word)`:  给定一个词，返回其中所有字符对的集合，这是BPE算法的核心部分。

**如何使用:** 这些函数通常在 `SimpleTokenizer` 类中使用，用于初始化和执行BPE编码。 他们很少直接被用户调用。

**2. 文本清洗相关的函数**

```python
def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = " ".join(text.split())
    text = text.strip()
    return text


def _clean_canonicalize(x):
    # basic, remove whitespace, remove punctuation, lower case
    return canonicalize_text(basic_clean(x))


def _clean_lower(x):
    # basic, remove whitespace, lower case
    return whitespace_clean(basic_clean(x)).lower()


def _clean_whitespace(x):
    # basic, remove whitespace
    return whitespace_clean(basic_clean(x))


def get_clean_fn(type: str):
    if type == 'canonicalize':
        return _clean_canonicalize
    elif type == 'lower':
        return _clean_lower
    elif type == 'whitespace':
        return _clean_whitespace
    else:
        assert False, f"Invalid clean function ({type})."


def canonicalize_text(
    text,
    *,
    keep_punctuation_exact_string=None,
    trans_punctuation: dict = str.maketrans("", "", string.punctuation),
):
    """Returns canonicalized `text` (lowercase and punctuation removed)."""
    text = text.replace("_", " ")
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(trans_punctuation)
            for part in text.split(keep_punctuation_exact_string)
        )
    else:
        text = text.translate(trans_punctuation)
    text = text.lower()
    text = " ".join(text.split())
    return text.strip()
```

**描述:** 这些函数用于文本的预处理，包括修复文本，移除HTML实体，标准化空格，移除标点符号以及将文本转换为小写。

- `basic_clean(text)`: 使用`ftfy`修复文本错误，并移除HTML实体。
- `whitespace_clean(text)`: 将文本中的多个空格替换为单个空格，并移除首尾空格。
- `canonicalize_text(text)`: 将文本转换为小写，并移除标点符号。
- `get_clean_fn(type: str)`:  根据传入的`type`参数，返回相应的文本清洗函数。
- `_clean_canonicalize(x)`, `_clean_lower(x)`, `_clean_whitespace(x)`: 是一些辅助函数，用于组合不同的清洗步骤。

**如何使用:** 这些函数通常在`SimpleTokenizer`类中使用，用于在tokenization之前对输入文本进行清洗，提高tokenization的准确性。  例如：

```python
text = "  Hello, world! &amp;  "
cleaned_text = basic_clean(text)
print(f"Basic Cleaned: {cleaned_text}") # 输出: Hello, world! &

cleaned_text = canonicalize_text(text)
print(f"Canonicalized: {cleaned_text}") # 输出: hello world
```

**3. `SimpleTokenizer` 类**

```python
import torch
import regex as re

class SimpleTokenizer(object):
    def __init__(
            self,
            bpe_path: str = default_bpe(),
            additional_special_tokens: Optional[List[str]] = None,
            context_length: Optional[int] = 77,
            clean: str = 'lower',
            reduction_mask: str = ''
    ):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        special_tokens = ['<start_of_text>', '<end_of_text>']
        if additional_special_tokens:
            special_tokens += additional_special_tokens
        vocab.extend(special_tokens)
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {t:t for t in special_tokens}
        special = "|".join(special_tokens)
        self.pat = re.compile(
            special + r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )
        self.vocab_size = len(self.encoder)
        self.all_special_ids = [self.encoder[t] for t in special_tokens]
        self.sot_token_id = self.all_special_ids[0]
        self.eot_token_id = self.all_special_ids[1]
        self.context_length = context_length
        self.clean_fn = get_clean_fn(clean)
        self.reduction_fn = get_reduction_mask_fn(reduction_mask) if reduction_mask else None

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except Exception:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = self.clean_fn(text)
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text

    def __call__(self, texts: Union[str, List[str]], context_length: Optional[int] = None) -> torch.LongTensor:
        """ Returns the tokenized representation of given input string(s)"""
        if isinstance(texts, str):
            texts = [texts]

        context_length = context_length or self.context_length
        assert context_length, 'Please set a valid context length'

        if self.reduction_fn is not None:
            # use reduction strategy for tokenize if set, otherwise default to truncation below
            return self.reduction_fn(
                texts,
                context_length=context_length,
                sot_token_id=self.sot_token_id,
                eot_token_id=self.eot_token_id,
                encode_fn=self.encode,
            )

        all_tokens = [[self.sot_token_id] + self.encode(text) + [self.eot_token_id] for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                tokens = tokens[:context_length]  # Truncate
                tokens[-1] = self.eot_token_id
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result
```

**描述:**  `SimpleTokenizer` 类是CLIP模型的tokenizer的核心。 它使用BPE算法将文本转换为tokens，并且包含文本清洗和截断等功能。

- `__init__`:  初始化tokenizer。加载BPE词汇表，创建encoder和decoder，以及编译正则表达式。
- `bpe(token)`:  对单个token应用BPE算法。
- `encode(text)`: 将文本编码为tokens。首先清洗文本，然后使用BPE算法将文本转换为tokens。
- `decode(tokens)`: 将tokens解码为文本。
- `__call__(texts)`:  tokenize一个或多个文本字符串。它将文本清洗、编码、截断/padding到指定的上下文长度。如果设置了`reduction_fn`, 则使用它来进行token处理, 否则进行简单截断。

**如何使用:**  
```python
tokenizer = SimpleTokenizer()
text = "This is an example sentence."
tokens = tokenizer(text)
print(f"Tokens: {tokens}")
decoded_text = tokenizer.decode(tokens[0].tolist()) # decode 需要list of int
print(f"Decoded Text: {decoded_text}")
```

**4.  `random_mask_tokenize`, `simple_mask_tokenize`, `syntax_mask_tokenize` 和 `get_reduction_mask_fn` 函数**

```python
import torch
import random
import nltk
import numpy as np
from typing import Callable, List, Union
from functools import partial

_nltk_init = False # 初始化nltk标志

def random_mask_tokenize(
        texts: Union[str, List[str]],
        context_length: int,
        sot_token_id: int,
        eot_token_id: int,
        encode_fn: Callable,
        shuffle: bool = False,
):
    all_tokens = [encode_fn(text) for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        tokens = torch.tensor(tokens)
        num_tokens = len(tokens)
        if num_tokens > context_length - 2:  # 2 for sot and eot token
            num_keep = context_length - 2
            indices = torch.randperm(len(tokens))
            indices = indices[:num_keep]
            if not shuffle:
                indices = indices.msort()
            tokens = tokens[indices]
            num_tokens = num_keep
        result[i, 0] = sot_token_id
        result[i, 1:num_tokens + 1] = tokens
        result[i, num_tokens + 1] = eot_token_id

    return result


def simple_mask_tokenize(
        texts: Union[str, List[str]],
        context_length: int,
        sot_token_id: int,
        eot_token_id: int,
        encode_fn: Callable,
):
    all_tokens = [encode_fn(text) for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        num_tokens = len(tokens)
        if num_tokens > context_length - 2:  # 2 for sot and eot token
            num_keep = context_length - 2
            start_index = random.randint(0, num_tokens - num_keep)  # high is incl
            tokens = tokens[start_index: start_index + num_keep]
        tokens = [sot_token_id] + tokens + [eot_token_id]
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def syntax_mask_tokenize(
        texts: Union[str, List[str]],
        context_length: int,
        sot_token_id: int,
        eot_token_id: int,
        encode_fn: Callable,
) -> torch.LongTensor:
    """ Returns the tokenized representation of given input string(s).
    Apply syntax masking before tokenize.
    """
    import nltk
    global _nltk_init
    if not _nltk_init:
        # run them for the first time
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        _nltk_init = True

    def get_order(x):
        if x.startswith('NN'):
            return 1
        elif x.startswith('JJ'):
            return 2
        elif x.startswith('VB'):
            return 3
        else:
            return 4

    # syntax masking
    new_texts = []
    for text in texts:
        list_tokens = nltk.tokenize.word_tokenize(text)
        pos_tags = nltk.pos_tag(list_tokens)
        #  sample the words by get_order method
        order_list = [get_order(tag) for _, tag in pos_tags]
        sorted_ids = np.argsort(np.array(order_list))
        sampled_ids = sorted(sorted_ids[:context_length - 2]) # need 2 slots for sot and eot tokens
        sampled_tokens = np.take(np.array(list_tokens), sampled_ids, axis=0)  # sample the tokens

        new_text = ''
        for token in sampled_tokens:
            new_text = new_text + str(token) + ' '
        new_text = new_text.strip()
        new_texts.append(new_text)
    texts = new_texts

    all_tokens = [[sot_token_id] + encode_fn(text) + [eot_token_id] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        # still need first truncate because some words produces two tokens
        if len(tokens) > context_length:
            tokens = tokens[:context_length]  # Truncate
            tokens[-1] = eot_token_id
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def get_reduction_mask_fn(type: str):
    """ Choose strategy for dropping (masking) tokens to achieve target context length"""
    assert type in ('simple', 'random', 'shuffle', 'syntax')
    if type == 'simple':
        return simple_mask_tokenize  # randomly select block [start:end]
    elif type == 'random':
        return random_mask_tokenize  # randomly drop tokens (keep order)
    elif type == 'shuffle':
        return partial(random_mask_tokenize, shuffle=True)  # randomly drop tokens (shuffle order)
    elif type == 'syntax':
        return syntax_mask_tokenize  # randomly drop prioritized by syntax
```

**描述:** 这些函数定义了不同的token masking策略，用于处理文本长度超过上下文长度的情况。

- `random_mask_tokenize(texts, context_length, sot_token_id, eot_token_id, encode_fn, shuffle=False)`: 随机选择tokens，保留`context_length - 2`个token，并添加`sot_token_id`和`eot_token_id`。`shuffle`参数控制是否打乱token的顺序。
- `simple_mask_tokenize(texts, context_length, sot_token_id, eot_token_id, encode_fn)`: 随机选择一个起始位置，然后从该位置开始截取`context_length - 2`个token。
- `syntax_mask_tokenize(texts, context_length, sot_token_id, eot_token_id, encode_fn)`:  使用nltk进行语法分析，根据词性选择tokens。首先选择名词，然后是形容词，然后是动词，最后是其他词性的词。
- `get_reduction_mask_fn(type)`: 根据`type`参数返回相应的masking函数。

**如何使用:** 这些函数通过`SimpleTokenizer`的`reduction_fn`参数来设置。  例如:

```python
tokenizer = SimpleTokenizer(reduction_mask='random') # 使用随机masking
text = "This is a very long example sentence that exceeds the context length."
tokens = tokenizer(text)
print(f"Tokens with random masking: {tokens}")
```

**5.  `HFTokenizer` 和 `SigLipTokenizer` 类**

```python
import torch
import warnings
from typing import Union, List, Optional

class HFTokenizer:
    """HuggingFace tokenizer wrapper"""

    def __init__(
            self,
            tokenizer_name: str,
            context_length: Optional[int] = 77,
            clean: str = 'whitespace',
            strip_sep_token: bool = False,
            language: Optional[str] = None,
            cache_dir: Optional[str] = None,
            **kwargs
    ):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir, **kwargs)
        set_lang_fn = getattr(self.tokenizer, 'set_src_lang_special_tokens', None)
        if callable(set_lang_fn):
            self.set_lang_fn = set_lang_fn
        if language is not None:
            self.set_language(language)
        self.context_length = context_length
        self.clean_fn = get_clean_fn(clean)
        self.strip_sep_token = strip_sep_token

    def save_pretrained(self, dest):
        self.tokenizer.save_pretrained(dest)

    def __call__(self, texts: Union[str, List[str]], context_length: Optional[int] = None) -> torch.Tensor:
        # same cleaning as for default tokenizer, except lowercasing
        # adding lower (for case-sensitive tokenizers) will make it more robust but less sensitive to nuance
        if isinstance(texts, str):
            texts = [texts]

        context_length = context_length or self.context_length
        assert context_length, 'Please set a valid context length in class init or call.'

        texts = [self.clean_fn(text) for text in texts]
        input_ids = self.tokenizer.batch_encode_plus(
            texts,
            return_tensors='pt',
            max_length=context_length,
            padding='max_length',
            truncation=True,
        ).input_ids

        if self.strip_sep_token:
            input_ids = torch.where(
                input_ids == self.tokenizer.sep_token_id,
                torch.zeros_like(input_ids),
                input_ids,
            )

        return input_ids
    
    def set_language(self, src_lang):
        if hasattr(self, 'set_lang_fn'):
            self.set_lang_fn(src_lang)
        else:
            warnings.warn('Cannot set language for the tokenizer.')


class SigLipTokenizer:
    """HuggingFace tokenizer wrapper for SigLIP T5 compatible sentencepiece vocabs"""

    VOCAB_FILES = {
        # english, vocab_size=32_000
        "c4-en": "http://storage.googleapis.com/t5-data/vocabs/cc_en.32000/sentencepiece.model",
        # used in multilingual models (mT5, PaLI), vocab_size=250_000
        "mc4": "http://storage.googleapis.com/t5-data/vocabs/mc4.250000.100extra/sentencepiece.model",
    }

    def __init__(
            self,
            tokenizer_name: str,
            context_length: Optional[int] = 64,
    ):
        from transformers import T5TokenizerFast

        if tokenizer_name in self.VOCAB_FILES:
            # FIXME temporary hack?
            import tempfile

            import fsspec
            vocab_file = self.VOCAB_FILES[tokenizer_name]
            with tempfile.NamedTemporaryFile('wb') as dst:
                with fsspec.open(vocab_file, 'rb') as src:
                    dst.write(src.read())
                self.tokenizer = T5TokenizerFast(dst.name, legacy=False)
        else:
            self.tokenizer = T5TokenizerFast(tokenizer_name, legacy=False)

        self.tokenizer.pad_token_id = 1
        self.tokenizer.eos_token_id = 1
        self.context_length = context_length

    def save_pretrained(self, dest):
        self.tokenizer.save_pretrained(dest)

    def __call__(self, texts: Union[str, List[str]], context_length: Optional[int] = None) -> torch.Tensor:
        # same cleaning as for default tokenizer, except lowercasing
        # adding lower (for case-sensitive tokenizers) will make it more robust but less sensitive to nuance
        if isinstance(texts, str):
            texts = [texts]

        context_length = context_length or self.context_length
        assert context_length, 'Please set a valid context length in class init or call.'

        texts = [canonicalize_text(basic_clean(text)) for text in texts]
        output = self.tokenizer(
            texts,
            return_tensors='pt',
            max_length=context_length,
            padding='max_length',
            truncation=True,
        )
        return output.input_ids
```

**描述:** 这些类是Hugging Face `transformers`库中tokenizers的包装器。

- `HFTokenizer`: 使用Hugging Face `AutoTokenizer`。 允许使用各种预训练的tokenizers，例如BERT，GPT-2等。它接受一个 `tokenizer_name` 参数，用于指定要使用的tokenizer。
- `SigLipTokenizer`:  是`T5TokenizerFast`的包装器，专门用于SigLIP模型。

**如何使用:**
```python
from transformers import logging
logging.set_verbosity_error() # suppress warnings of HFTokenizer

tokenizer = HFTokenizer(tokenizer_name='bert-base-uncased')
text = "This is a sentence to be tokenized."
tokens = tokenizer(text)
print(f"HFTokenizer Tokens: {tokens}")
```

总体来说，这段代码提供了一个灵活且可扩展的文本tokenization工具集，能够支持不同的BPE词汇表，文本清洗策略，token masking策略以及来自Hugging Face Transformers的预训练tokenizers。