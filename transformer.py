import os
import re
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import jieba
from collections import Counter

# 配置
warnings.filterwarnings('ignore')  # 忽略警告
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 文本预处理类
class TextPreprocessor:
    def __init__(self, max_features=10000, max_len=128, stopwords_file=None):
        self.max_features = max_features  # 最大词汇数
        self.max_len = max_len  # 最大文本长度
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}  # 初始化字典，<PAD>和<UNK>为特殊字符
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}  # 反向字典
        self.word_counts = Counter()  # 词频统计

        self.stopwords = set()  # 停用词
        if stopwords_file:
            # 如果提供了停用词文件，读取并存入
            with open(stopwords_file, encoding='utf-8') as f:
                self.stopwords = set(f.read().strip().splitlines())

    def clean_text(self, text):
        """清理文本"""
        text = re.sub(r'[^A-Za-z0-9\u4e00-\u9fa5]+', ' ', text)  # 去掉非字母、数字、汉字的字符
        text = text.strip()  # 去除首尾空格
        return text

    def tokenize(self, text):
        """分词并去除停用词"""
        text = self.clean_text(text)  # 清理文本
        words = jieba.lcut(str(text))  # 使用jieba进行分词
        words = [word for word in words if word not in self.stopwords]  # 去除停用词
        return words

    def fit(self, texts):
        """构建词典"""
        for text in texts:
            self.word_counts.update(self.tokenize(text))  # 统计词频
        for word, _ in self.word_counts.most_common(self.max_features - 2):  # 只保留top N个词
            idx = len(self.word2idx)
            self.word2idx[word] = idx  # 为每个词分配索引
            self.idx2word[idx] = word  # 更新索引与词的反向映射

    def text_to_sequence(self, text):
        """将文本转为序列"""
        words = self.tokenize(text)
        sequence = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]  # 将词转换为索引
        # 如果文本短于最大长度，补充< PAD >，如果超出则截断
        if len(sequence) < self.max_len:
            sequence = sequence + [self.word2idx['<PAD>']] * (self.max_len - len(sequence))
        else:
            sequence = sequence[:self.max_len]
        return sequence


# 数据集类
class TextDataset(Dataset):
    def __init__(self, texts, labels, preprocessor):
        self.texts = [preprocessor.text_to_sequence(text) for text in texts]  # 将文本转为序列
        self.labels = labels  # 标签

    def __len__(self):
        return len(self.texts)  # 返回数据集大小

    def __getitem__(self, idx):
        """获取数据集中一个样本"""
        return {
            'input_ids': torch.tensor(self.texts[idx], dtype=torch.long),
            'label': torch.tensor(self.labels[idx] if self.labels is not None else -1, dtype=torch.long)
        }


# Transformer编码器层
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, max_len=128):
        super().__init__()
        # 定义多头自注意力机制
        self.d_model = d_model
        self.n_heads = n_heads
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_ff)  # 前馈神经网络的第一层
        self.dropout = nn.Dropout(dropout)  # Dropout层
        self.linear2 = nn.Linear(d_ff, d_model)  # 前馈神经网络的第二层
        self.norm1 = nn.LayerNorm(d_model)  # 层归一化
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()  # 激活函数

    def forward(self, src):
        # 自注意力机制部分
        attn_output, attn_weights = self.self_attn(src, src, src)
        src = src + self.dropout1(attn_output)  # 残差连接
        src = self.norm1(src)  # 层归一化
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))  # 前馈神经网络
        src = src + self.dropout2(ff_output)  # 残差连接
        src = self.norm2(src)  # 层归一化
        return src


# Transformer模型类
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, num_layers, num_classes, max_len=128, dropout=0.1):
        super().__init__()
        # 初始化embedding层
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, d_model))  # 位置编码
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, max_len)
            for _ in range(num_layers)
        ])  # 堆叠多层编码器
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)  # 输出层

    def forward(self, src):
        embedded = self.embedding(src)  # 将词索引转为词向量
        embedded = embedded + self.positional_encoding[:, :src.size(1), :]  # 添加位置编码
        embedded = self.dropout(embedded)
        embedded = embedded.permute(1, 0, 2)  # 调整维度，适应多头自注意力层
        for layer in self.encoder_layers:  # 通过每一层编码器
            embedded = layer(embedded)
        embedded = embedded.permute(1, 0, 2)
        pooled = embedded.mean(dim=1)  # 对所有位置的隐藏状态进行平均池化
        out = self.fc(pooled)  # 通过全连接层得到输出
        return out


# 分类器类
class TransformerClassifier:
    def __init__(self, d_model=128, n_heads=8, d_ff=256, num_layers=2, dropout=0.1,
                 batch_size=32, max_features=10000, max_len=128, learning_rate=1e-3,
                 epochs=10, stopwords_file=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 判断是否使用GPU
        self.preprocessor = TextPreprocessor(max_features=max_features, max_len=max_len, stopwords_file=stopwords_file)
        self.batch_size = batch_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_len = max_len
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.label_encoder = LabelEncoder()  # 标签编码器
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}  # 存储训练过程中的历史数据
        self.best_val_acc = 0  # 最佳验证准确率

    def prepare_data(self, data_dir):
        """加载和准备数据"""
        texts = []
        labels = []
        label_names = []  # 保存唯一的文件夹名称

        # 遍历数据集文件夹
        for folder in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder)
            if os.path.isdir(folder_path):
                label_names.append(folder)  # 记录类别名称
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)
                    with open(file_path, encoding='gbk', errors='ignore') as f:
                        texts.append(f.read())  # 读取文本内容
                        labels.append(folder)  # 使用文件夹名称作为标签

        # 标签编码：对唯一的文件夹名称进行编码
        self.label_encoder.fit(label_names)
        encoded_labels = self.label_encoder.transform(labels)

        # 构建词典
        self.preprocessor.fit(texts)
        vocab_size = len(self.preprocessor.word2idx)

        # 划分数据集
        X_train, X_temp, y_train, y_temp = train_test_split(
            texts, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        # 创建数据加载器
        self.train_loader = DataLoader(
            TextDataset(X_train, y_train, self.preprocessor),
            batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            TextDataset(X_val, y_val, self.preprocessor),
            batch_size=self.batch_size
        )
        self.test_loader = DataLoader(
            TextDataset(X_test, y_test, self.preprocessor),
            batch_size=self.batch_size
        )

        num_classes = len(self.label_encoder.classes_)

        # 初始化模型
        self.model = TransformerModel(
            vocab_size=vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            num_layers=self.num_layers,
            num_classes=num_classes,
            max_len=self.max_len,
            dropout=self.dropout
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()  # 损失函数
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)  # 优化器

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch in tqdm(self.train_loader, desc='训练中'):
            self.optimizer.zero_grad()  # 清空梯度
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['label'].to(self.device)
            outputs = self.model(input_ids)  # 模型输出
            loss = self.criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            self.optimizer.step()  # 更新参数
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)  # 获取预测结果
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return total_loss / len(self.train_loader), correct / total  # 返回平均损失和准确率

    def evaluate(self, data_loader):
        """评估模型表现"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []
        with torch.no_grad():  # 不计算梯度
            for batch in tqdm(data_loader, desc='评估中'):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.model(input_ids)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
        return (
            total_loss / len(data_loader),
            correct / total,
            all_preds,
            all_labels,
            np.array(all_probs)
        )

    def train_model(self, save_path='best_transformer_model.pt'):
        """训练模型"""
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print("-" * 10)
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, _, _, _ = self.evaluate(self.val_loader)
            print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
            print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(self.model.state_dict(), save_path)
                print(f"最佳模型已保存至 {save_path}")

    def plot_training_history(self):
        """绘制训练过程中的损失和准确率"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], marker='o', label='train loss')
        plt.plot(self.history['val_loss'], marker='o', label='val loss')
        plt.title('Transformer Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_acc'], marker='o', label='train acc')
        plt.plot(self.history['val_acc'], marker='o', label='val acc')
        plt.title('Transformer Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()

    def evaluate_and_plot(self, model_path='best_transformer_model.pt'):
        """评估并绘制测试结果"""
        self.model.load_state_dict(torch.load(model_path))  # 加载最佳模型
        self.model.to(self.device)
        self.model.eval()

        test_loss, test_acc, y_pred, y_true, test_probs = self.evaluate(self.test_loader)
        print(f"\n测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.4f}")

        # 转换回原始标签（文件夹名称）
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        y_true_labels = self.label_encoder.inverse_transform(y_true)

        # 分类报告，使用真实的标签名称
        print("\n分类报告:")
        print(classification_report(y_true_labels, y_pred_labels, digits=4, target_names=self.label_encoder.classes_))

        # 混淆矩阵，使用真实的标签名称
        cm = confusion_matrix(y_true_labels, y_pred_labels, labels=self.label_encoder.classes_)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Transformer Confusion Matrix')
        plt.show()

        # ROC曲线
        y_test_bin = label_binarize(y_true, classes=range(len(self.label_encoder.classes_)))
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), test_probs.ravel())
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})',
                 color='deeppink', linewidth=4)
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Transformer Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()


# 使用示例
if __name__ == '__main__':
    data_dir = 'cn_news'
    stopwords_file = '/root/autodl-tmp/test/baidu_stopwords.txt'

    classifier = TransformerClassifier(
        d_model=128,
        n_heads=8,
        d_ff=256,
        num_layers=2,
        dropout=0.1,
        batch_size=32,
        max_features=10000,
        max_len=128,
        learning_rate=1e-3,
        epochs=10,
        stopwords_file=stopwords_file
    )

    classifier.prepare_data(data_dir)
    classifier.train_model(save_path='best_transformer_model.pt')
    classifier.plot_training_history()
    classifier.evaluate_and_plot(model_path='best_transformer_model.pt')
