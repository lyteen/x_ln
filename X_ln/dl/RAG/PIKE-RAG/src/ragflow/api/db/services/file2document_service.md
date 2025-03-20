Lan: `py` From`dl/RAG/PIKE-RAG/src\ragflow\api\db\services\file2document_service.py`

好的，我将重新开始，并提供更清晰、更详细的解释和代码示例，并使用中文进行注释。

**1. `File2DocumentService` 类总览**

`File2DocumentService` 类负责处理 `File` 和 `Document` 之间的关联关系，它继承自 `CommonService`，并使用 `File2Document` 模型进行数据库操作。该类的作用是维护文件和文档之间的映射关系，方便根据文件或文档 ID 查找相关信息。

**2. 核心代码片段解释**

```python
from datetime import datetime

from api.db import FileSource
from api.db.db_models import DB
from api.db.db_models import File, File2Document
from api.db.services.common_service import CommonService
from api.db.services.document_service import DocumentService
from api.utils import current_timestamp, datetime_format


class File2DocumentService(CommonService):
    model = File2Document  # 指定该Service操作的数据库模型为File2Document

    @classmethod
    @DB.connection_context()  # 使用数据库连接上下文管理器，确保数据库连接的正确处理
    def get_by_file_id(cls, file_id):
        """根据文件ID获取 File2Document 对象"""
        objs = cls.model.select().where(cls.model.file_id == file_id)  # 构建查询，查找 file_id 匹配的记录
        return objs  # 返回查询结果

    @classmethod
    @DB.connection_context()
    def get_by_document_id(cls, document_id):
        """根据文档ID获取 File2Document 对象"""
        objs = cls.model.select().where(cls.model.document_id == document_id)  # 构建查询，查找 document_id 匹配的记录
        return objs

    @classmethod
    @DB.connection_context()
    def insert(cls, obj):
        """插入 File2Document 对象"""
        if not cls.save(**obj):  # 调用父类的 save 方法保存对象
            raise RuntimeError("Database error (File)!")  # 如果保存失败，抛出异常
        return File2Document(**obj)  # 返回新创建的 File2Document 对象

    @classmethod
    @DB.connection_context()
    def delete_by_file_id(cls, file_id):
        """根据文件ID删除 File2Document 对象"""
        return cls.model.delete().where(cls.model.file_id == file_id).execute()  # 构建删除查询，删除 file_id 匹配的记录

    @classmethod
    @DB.connection_context()
    def delete_by_document_id(cls, doc_id):
        """根据文档ID删除 File2Document 对象"""
        return cls.model.delete().where(cls.model.document_id == doc_id).execute()  # 构建删除查询，删除 document_id 匹配的记录

    @classmethod
    @DB.connection_context()
    def update_by_file_id(cls, file_id, obj):
        """根据文件ID更新 File2Document 对象"""
        obj["update_time"] = current_timestamp()  # 更新 update_time 字段
        obj["update_date"] = datetime_format(datetime.now())  # 更新 update_date 字段
        cls.model.update(obj).where(cls.model.id == file_id).execute()  # 构建更新查询，更新 file_id 匹配的记录
        return File2Document(**obj)  # 返回更新后的 File2Document 对象

    @classmethod
    @DB.connection_context()
    def get_storage_address(cls, doc_id=None, file_id=None):
        """获取文件或文档的存储地址"""
        if doc_id:
            f2d = cls.get_by_document_id(doc_id)  # 根据文档ID获取 File2Document 对象
        else:
            f2d = cls.get_by_file_id(file_id)  # 根据文件ID获取 File2Document 对象

        if f2d:
            file = File.get_by_id(f2d[0].file_id)  # 获取关联的文件对象
            if not file.source_type or file.source_type == FileSource.LOCAL:  # 如果文件源类型为本地或未指定
                return file.parent_id, file.location  # 返回文件的父ID和位置
            doc_id = f2d[0].document_id # 如果不是本地文件，则从File2Document中获取doc_id

        assert doc_id, "please specify doc_id"  # 如果没有提供doc_id，抛出断言错误
        e, doc = DocumentService.get_by_id(doc_id)  # 获取文档对象
        return doc.kb_id, doc.location  # 返回文档的知识库ID和位置
```

**3. 代码解释**

*   **`model = File2Document`**:  指定了该 `Service` 类操作的数据库模型为 `File2Document`。`File2Document` 模型定义了文件和文档之间关系的表结构。
*   **`@DB.connection_context()`**:  这是一个装饰器，用于管理数据库连接。它确保在方法执行前后正确地获取和释放数据库连接，防止连接泄漏和提高代码健壮性。
*   **`get_by_file_id(cls, file_id)` 和 `get_by_document_id(cls, document_id)`**:  这两个方法分别根据文件 ID 和文档 ID 查询 `File2Document` 对象。它们使用 `peewee` 提供的 `select()` 和 `where()` 方法构建 SQL 查询。
*   **`insert(cls, obj)`**:  该方法用于向数据库中插入新的 `File2Document` 对象。它首先调用父类的 `save()` 方法保存对象，如果保存失败，则抛出异常。
*   **`delete_by_file_id(cls, file_id)` 和 `delete_by_document_id(cls, doc_id)`**:  这两个方法分别根据文件 ID 和文档 ID 删除 `File2Document` 对象。
*   **`update_by_file_id(cls, file_id, obj)`**:  该方法用于根据文件 ID 更新 `File2Document` 对象。它首先更新 `update_time` 和 `update_date` 字段，然后使用 `peewee` 提供的 `update()` 和 `where()` 方法构建 SQL 更新查询。
*   **`get_storage_address(cls, doc_id=None, file_id=None)`**:  该方法用于获取文件或文档的存储地址。它首先尝试根据文件 ID 或文档 ID 获取 `File2Document` 对象。如果找到，则根据文件源类型返回文件的父 ID 和位置，或者根据文档 ID 获取文档对象并返回文档的知识库 ID 和位置。

**4. 使用场景示例**

假设我们有一个文件，其 ID 为 `file_id = 123`，并且该文件关联到一个文档，其 ID 为 `doc_id = 456`。

1.  **查询文件关联的文档：**

    ```python
    file2doc = File2DocumentService.get_by_file_id(file_id=123)
    if file2doc:
        print(f"文件 ID {123} 关联的文档 ID 为: {file2doc[0].document_id}")
    else:
        print(f"未找到文件 ID {123} 关联的文档")
    ```

2.  **查询文档关联的文件：**

    ```python
    file2doc = File2DocumentService.get_by_document_id(document_id=456)
    if file2doc:
        print(f"文档 ID {456} 关联的文件 ID 为: {file2doc[0].file_id}")
    else:
        print(f"未找到文档 ID {456} 关联的文件")
    ```

3.  **获取文件/文档的存储地址：**

    ```python
    kb_id, location = File2DocumentService.get_storage_address(doc_id=456)
    print(f"文档 ID {456} 的存储地址为：知识库 ID: {kb_id