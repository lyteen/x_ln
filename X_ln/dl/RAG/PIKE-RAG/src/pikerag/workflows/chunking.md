Lan: `py` From`dl/RAG/PIKE-RAG/src\pikerag\workflows\chunking.py`

好的，我将重新开始，并为代码的关键部分提供解释，生成小段代码并附带中文描述，同时包含代码的用法和简单的示例。

**1. `ChunkingWorkflow` 类**

```python
import importlib
import os
from typing import List, Tuple

import pickle
from tqdm import tqdm

from pikerag.document_loaders import get_loader
from pikerag.document_transformers import LLMPoweredRecursiveSplitter
from pikerag.llm_client import BaseLLMClient
from pikerag.utils.config_loader import load_class
from pikerag.utils.logger import Logger
from pikerag.utils.walker import list_files_recursively


class ChunkingWorkflow:
    def __init__(self, yaml_config: dict) -> None:
        self._yaml_config: dict = yaml_config

        self._init_logger()
        self._init_splitter()

        self._init_file_infos()
        return

    def run(self) -> None:
        for doc_name, input_path, output_path in tqdm(self._file_infos, desc="Chunking file"):
            if os.path.exists(output_path) is True:
                self._logger.info(f"Skip file: {doc_name} due to output already exist!")
                continue

            self._logger.info(f"Loading file: {doc_name}")

            # Try get the file loader and load documents
            doc_loader = get_loader(file_path=input_path, file_type=None)
            if doc_loader is None:
                self._logger.info(f"Skip file {doc_name} due to undefined Document Loader.")
                continue
            docs = doc_loader.load()

            # Add metadata
            for doc in docs:
                doc.metadata.update({"filename": doc_name})

            # Document Splitting
            chunk_docs = self._splitter.transform_documents(docs)

            # Dump document chunks to disk.
            with open(output_path, "wb") as fout:
                pickle.dump(chunk_docs, fout)
```

**描述:** `ChunkingWorkflow` 类是整个分块流程的核心。它接收一个 YAML 配置文件，并根据配置文件的设置来初始化日志记录器、分块器和文件信息。`run` 方法遍历所有输入文件，加载它们，将它们分割成块，并将结果保存到磁盘上。

**如何使用:** 首先，需要创建一个包含配置信息的 YAML 文件。然后，创建一个 `ChunkingWorkflow` 实例，并将 YAML 配置文件传递给它。最后，调用 `run` 方法来执行分块流程。

**2. `_init_logger` 方法**

```python
    def _init_logger(self) -> None:
        self._logger: Logger = Logger(
            name=self._yaml_config["experiment_name"],
            dump_folder=self._yaml_config["log_dir"],
        )
```

**描述:** `_init_logger` 方法初始化日志记录器。它使用配置文件中的实验名称和日志目录来创建一个 `Logger` 实例。这个 logger 用于记录流程中的各种信息，例如加载了哪些文件，跳过了哪些文件，以及发生了哪些错误。

**如何使用:**  这个方法由 `ChunkingWorkflow` 类的构造函数自动调用。你不需要直接调用它。日志将保存在配置文件中指定的目录中。

**3. `_init_llm_client` 方法**

```python
    def _init_llm_client(self) -> None:
        # Dynamically import the LLM client.
        self._client_logger = Logger(name="client", dump_mode="a", dump_folder=self._yaml_config["log_dir"])

        llm_client_config = self._yaml_config["llm_client"]
        cache_location = os.path.join(
            self._yaml_config["log_dir"],
            f"{llm_client_config['cache_config']['location_prefix']}.db",
        )

        client_module = importlib.import_module(llm_client_config["module_path"])
        client_class = getattr(client_module, llm_client_config["class_name"])
        assert issubclass(client_class, BaseLLMClient)
        self._client = client_class(
            location=cache_location,
            auto_dump=llm_client_config["cache_config"]["auto_dump"],
            logger=self._client_logger,
            llm_config=llm_client_config["llm_config"],
            **llm_client_config.get("args", {}),
        )
        return
```

**描述:** `_init_llm_client` 方法初始化 LLM 客户端。它动态地从配置文件中指定的模块导入 LLM 客户端类，并创建一个实例。该客户端用于与大型语言模型 (LLM) 进行交互，例如用于生成文本摘要或进行语义分割。 它同时初始化client logger，用于缓存和记录LLM交互过程。

**如何使用:**  这个方法由 `_init_splitter` 方法自动调用，前提是分块器需要使用 LLM。  你需要提供 LLM 客户端的配置信息，包括模块路径、类名、缓存设置和 LLM 配置。

**4. `_init_splitter` 方法**

```python
    def _init_splitter(self) -> None:
        splitter_config: dict = self._yaml_config["splitter"]
        splitter_args: dict = splitter_config.get("args", {})

        splitter_class = load_class(
            module_path=splitter_config["module_path"],
            class_name=splitter_config["class_name"],
            base_class=None,
        )

        if issubclass(splitter_class, (LLMPoweredRecursiveSplitter)):
            # Initialize LLM client
            self._init_llm_client()

            # Update args
            splitter_args["llm_client"] = self._client
            splitter_args["llm_config"] = self._yaml_config["llm_client"]["llm_config"]

            splitter_args["logger"] = self._logger

        if issubclass(splitter_class, LLMPoweredRecursiveSplitter):
            # Load protocols
            protocol_configs = self._yaml_config["chunking_protocol"]
            protocol_module = importlib.import_module(protocol_configs["module_path"])
            chunk_summary_protocol = getattr(protocol_module, protocol_configs["chunk_summary"])
            chunk_summary_refinement_protocol = getattr(protocol_module, protocol_configs["chunk_summary_refinement"])
            chunk_resplit_protocol = getattr(protocol_module, protocol_configs["chunk_resplit"])

            # Update args
            splitter_args["first_chunk_summary_protocol"] = chunk_summary_protocol
            splitter_args["last_chunk_summary_protocol"] = chunk_summary_refinement_protocol
            splitter_args["chunk_resplit_protocol"] = chunk_resplit_protocol

        self._splitter = splitter_class(**splitter_args)
        return
```

**描述:** `_init_splitter` 方法初始化文档分割器。它从配置文件中加载分割器类，并根据配置设置创建实例。如果分割器是 `LLMPoweredRecursiveSplitter` 的子类，它还会初始化 LLM 客户端，并将客户端传递给分割器。这段代码根据配置文件，选择合适的文档分割方法。 如果使用 LLM 支持的分割器，例如可以根据文本内容进行智能分割，则需要初始化 LLM client。

**如何使用:** 这个方法由 `ChunkingWorkflow` 类的构造函数自动调用。你需要在配置文件中指定分割器的模块路径、类名和参数。

**5. `_init_file_infos` 方法**

```python
    def _init_file_infos(self) -> None:
        input_setting: dict = self._yaml_config.get("input_doc_setting")
        output_setting: dict = self._yaml_config.get("output_doc_setting")
        assert input_setting is not None and output_setting is not None, (
            f"input_doc_setting and output_doc_setting should be provided!"
        )

        input_file_infos = list_files_recursively(
            directory=input_setting.get("doc_dir"),
            extensions=input_setting.get("extensions"),
        )

        output_dir = output_setting.get("doc_dir")
        output_suffix = output_setting.get("suffix", "pkl")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        self._file_infos: List[Tuple[str, str, str]] = [
            (doc_name, doc_path, os.path.join(output_dir, f"{os.path.splitext(doc_name)[0]}.{output_suffix}"))
            for doc_name, doc_path in input_file_infos
        ]
        return
```

**描述:** `_init_file_infos` 方法初始化文件信息列表。它从配置文件中读取输入和输出文档的设置，并使用 `list_files_recursively` 函数递归地列出输入目录中的所有文件。然后，它创建一个包含文档名称、输入路径和输出路径的元组列表。 这个函数用于收集需要处理的文件信息，并准备好输出路径。

**如何使用:** 这个方法由 `ChunkingWorkflow` 类的构造函数自动调用。你需要在配置文件中指定输入和输出文档的设置，包括文档目录和文件扩展名。

**6. `run` 方法**

```python
    def run(self) -> None:
        for doc_name, input_path, output_path in tqdm(self._file_infos, desc="Chunking file"):
            if os.path.exists(output_path) is True:
                self._logger.info(f"Skip file: {doc_name} due to output already exist!")
                continue

            self._logger.info(f"Loading file: {doc_name}")

            # Try get the file loader and load documents
            doc_loader = get_loader(file_path=input_path, file_type=None)
            if doc_loader is None:
                self._logger.info(f"Skip file {doc_name} due to undefined Document Loader.")
                continue
            docs = doc_loader.load()

            # Add metadata
            for doc in docs:
                doc.metadata.update({"filename": doc_name})

            # Document Splitting
            chunk_docs = self._splitter.transform_documents(docs)

            # Dump document chunks to disk.
            with open(output_path, "wb") as fout:
                pickle.dump(chunk_docs, fout)
```

**描述:**  `run` 方法是整个分块流程的入口点。它遍历 `_file_infos` 列表中的每个文件，并执行以下步骤：

1.  **检查输出文件是否存在:** 如果输出文件已经存在，则跳过该文件。
2.  **加载文档:** 使用 `get_loader` 函数根据文件路径加载文档。
3.  **添加元数据:** 将文件名添加到文档的元数据中。
4.  **分割文档:** 使用 `_splitter` 实例将文档分割成块。
5.  **保存文档块:** 将文档块保存到磁盘上的输出文件中。

**如何使用:**  在创建 `ChunkingWorkflow` 实例后，调用 `run` 方法来启动分块流程。 它会读取配置文件，读取文件，分割文件，然后将分割好的文件输出到指定目录。

**总体来说，这段代码实现了一个文档分块的pipeline, 它可以从指定目录读取文件，然后根据配置文件的参数，将文件分割成指定的块，然后存储到指定目录。**
