## Abstract

**Keywords:** Ensemble Learning, Deep Learning, Fake News Detection, FastText
<details>
    <summary>关键词</summary>
    <ul>
        集成学习, 深度学习, 假新闻检测, FastText
    <ul>
</details>

**Abstract:**
This research investigates machine learning, deep learning, and ensemble learning techniques for Arabic fake news detection. Integrating FastText word embeddings with various ML and DL methods, the study leverages advanced transformer-based models (BERT, XLNet, RoBERTa) optimized through hyperparameter tuning. Using AFND and ARABICFAKETWEETS datasets, hybrid DL models (CNN-LSTM, RNN-CNN, RNN-LSTM, Bi-GRU-Bi-LSTM) are presented, with the Bi-GRU-Bi-LSTM model achieving superior F1 score, accuracy, and loss metrics. The Bi-GRU-Bi-LSTM model attains 0.97 precision, 0.97 recall, 0.98 F1 score, and 0.98 accuracy on AFND, and 0.98 precision, 0.98 recall, 0.99 F1 score, and 0.99 accuracy on ARABICFAKETWEETS. The study concludes Bi-GRU-Bi-LSTM significantly outperforms in Arabic fake news detection, contributing to the fight against disinformation and setting the stage for multilingual fake news detection research.

<details>
    <summary>摘要</summary>
    <ul>
        本研究调查了用于阿拉伯语假新闻检测的机器学习、深度学习和集成学习技术。该研究将FastText词嵌入与各种ML和DL方法相结合，利用通过超参数调整优化的先进的基于transformer的模型（BERT、XLNet、RoBERTa）。使用AFND和ARABICFAKETWEETS数据集，提出了混合DL模型（CNN-LSTM、RNN-CNN、RNN-LSTM、Bi-GRU-Bi-LSTM），其中Bi-GRU-Bi-LSTM模型在F1分数、准确性和损失指标方面表现出色。Bi-GRU-Bi-LSTM模型在AFND数据集上达到0.97的精度、0.97的召回率、0.98的F1分数和0.98的准确率，在ARABICFAKETWEETS数据集上达到0.98的精度、0.98的召回率、0.99的F1分数和0.99的准确率。研究得出结论，Bi-GRU-Bi-LSTM在阿拉伯语假新闻检测方面显著优于其他模型，有助于打击虚假信息，并为多语言假新闻检测研究奠定了基础。
    <ul>
</details>

**Main Methods:**

1.  **FastText Word Embeddings:**  Utilized for efficient text data processing in conjunction with machine learning and deep learning models.
2.  **Machine Learning Models:** Employed Decision Tree, Support Vector Machine, Random Forest, Logistic Regression, and CATBoost.
3.  **Deep Learning Models:**  Used LSTM, GRU, and CNN to capture contextual information and sequential dependencies.
4.  **Transformer-Based Models:**  Leveraged BERT, XLNet, and RoBERTa for text categorization, optimized through hyperparameter tuning.
5.  **Ensemble Learning:** Hybrid deep learning models such as CNN-LSTM, RNN-CNN, RNN-LSTM, and Bi-GRU-Bi-LSTM were developed.

<details>
    <summary>主要方法</summary>
        <ul>
            <li><strong>FastText词嵌入：</strong> 与机器学习和深度学习模型结合使用，以实现高效的文本数据处理。</li>
            <li><strong>机器学习模型：</strong> 采用了决策树、支持向量机、随机森林、逻辑回归和CATBoost。</li>
            <li><strong>深度学习模型：</strong> 使用了LSTM、GRU和CNN来捕获上下文信息和序列依赖性。</li>
            <li><strong>基于Transformer的模型：</strong> 利用BERT、XLNet和RoBERTa进行文本分类，并通过超参数调整进行了优化。</li>
            <li><strong>集成学习：</strong> 开发了混合深度学习模型，如CNN-LSTM、RNN-CNN、RNN-LSTM和Bi-GRU-Bi-LSTM。</li>
        </ul>
</details>

**Main Contributions:**

1.  **Multilayer Preprocessing Framework:** Introduced using fake and real news text datasets, with NLP techniques to prepare data for word embeddings.
2.  **Integration of FastText Embeddings:** Incorporated both supervised and unsupervised FastText embeddings into machine learning and deep learning models.
3.  **Hybrid Deep Learning Models:** Developed and presented four hybrid deep learning models: CNN-LSTM, RNN-CNN, RNN-LSTM, and Bi-GRU-Bi-LSTM.
4.  **Superior Performance:** Demonstrated that the Bi-GRU-Bi-LSTM model significantly outperforms other models in Arabic fake news detection.

<details>
    <summary>主要贡献</summary>
        <ul>
            <li><strong>多层预处理框架：</strong> 引入了使用虚假和真实新闻文本数据集的框架，并使用NLP技术来准备用于词嵌入的数据。</li>
            <li><strong>FastText嵌入的集成：</strong> 将监督和非监督的FastText嵌入都纳入了机器学习和深度学习模型中。</li>
            <li><strong>混合深度学习模型：</strong> 开发并展示了四种混合深度学习模型：CNN-LSTM、RNN-CNN、RNN-LSTM和Bi-GRU-Bi-LSTM。</li>
            <li><strong>卓越的性能：</strong> 证明了Bi-GRU-Bi-LSTM模型在阿拉伯语假新闻检测方面明显优于其他模型。</li>
        </ul>
</details>
