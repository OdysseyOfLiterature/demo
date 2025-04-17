import os
import re
from colorama import Fore
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

# 配置
warnings.filterwarnings('ignore')  # 忽略警告信息
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体支持
plt.rcParams['axes.unicode_minus'] = False  # 防止负号显示问题


# 数据集类
class CommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, stopwords, max_len=128):
        """
        初始化数据集类，包括输入文本、标签、tokenizer、停用词和最大长度
        """
        self.tokenizer = tokenizer  # 使用的BERT tokenizer
        self.texts = texts  # 文本数据
        self.labels = labels  # 标签
        self.stopwords = stopwords  # 停用词
        self.max_len = max_len  # 文本最大长度

    def __len__(self):
        """返回数据集的大小"""
        return len(self.texts)

    def preprocess_text(self, text):
        """
        对文本进行预处理，包括去除停用词和特殊字符
        """
        text = re.sub(r'[^A-Za-z0-9\u4e00-\u9fa5]+', ' ', text)  # 只保留中文、英文和数字字符
        text = text.strip()  # 去除两端空白字符
        words = text.split()  # 以空格为分隔符分词
        words = [word for word in words if word not in self.stopwords]  # 去除停用词
        return ' '.join(words)  # 返回清洗后的文本

    def __getitem__(self, idx):
        """
        获取数据集中的一个样本，进行文本预处理并返回相关张量
        """
        text = self.texts[idx]  # 获取文本
        text = self.preprocess_text(text)  # 进行预处理

        # 对文本进行tokenize，并返回input_ids、attention_mask和标签
        encoding = self.tokenizer.encode_plus(
            str(text),
            max_length=self.max_len,  # 截断或填充至最大长度
            padding='max_length',  # 填充至最大长度
            truncation=True,  # 截断
            return_attention_mask=True,  # 返回attention mask
            return_tensors='pt'  # 返回Pytorch tensor格式
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),  # 返回flatten的input_ids
            'attention_mask': encoding['attention_mask'].flatten(),  # 返回flatten的attention_mask
            'labels': torch.tensor(self.labels[idx] if self.labels is not None else -1, dtype=torch.long)  # 返回标签
        }


# BERT-CNN模型定义
class BERTCNNClassifier(nn.Module):
    def __init__(self, bert_model_name, num_labels, hidden_size=768, dropout=0.1):
        """
        初始化BERT-CNN模型，BERT用于提取特征，CNN用于进行进一步的特征学习
        """
        super(BERTCNNClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)  # 加载BERT预训练模型
        self.conv = nn.Conv1d(in_channels=hidden_size, out_channels=256, kernel_size=3)  # 定义1D卷积层
        self.relu = nn.ReLU()  # ReLU激活函数
        self.pool = nn.MaxPool1d(kernel_size=2)  # 最大池化层
        self.fc = nn.Linear(256 * ((128 - 3 + 1) // 2), num_labels)  # 全连接层
        self.dropout = nn.Dropout(dropout)  # Dropout层，防止过拟合

    def forward(self, input_ids, attention_mask):
        """
        前向传播函数
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)  # BERT模型的输出
        sequence_output = outputs.last_hidden_state  # 提取BERT的最后一层隐藏状态
        x = sequence_output.transpose(1, 2)  # 转置维度以适应CNN的输入
        x = self.conv(x)  # 卷积操作
        x = self.relu(x)  # ReLU激活
        x = self.pool(x)  # 池化操作
        x = x.view(x.size(0), -1)  # 展平卷积层输出
        x = self.dropout(x)  # Dropout层
        return self.fc(x)  # 返回经过全连接层的输出


# 分类器封装类
class BERTCNNClassifierWrapper:
    def __init__(self, model_name='bert-base-chinese', num_labels=None, batch_size=32, max_len=128,
                 stopwords_file=None):
        """
        初始化BERT-CNN分类器封装类
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 判断是否使用GPU
        self.tokenizer = BertTokenizer.from_pretrained(model_name)  # 加载BERT tokenizer
        self.batch_size = batch_size  # 设置批量大小
        self.max_len = max_len  # 设置最大长度
        self.model_name = model_name  # 设置模型名称
        self.num_labels = num_labels  # 设置标签数量
        self.label_encoder = LabelEncoder()  # 标签编码器，用于将标签转化为数字
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}  # 记录训练历史
        self.best_val_acc = 0  # 保存最佳验证准确率

        # 加载停用词
        self.stopwords = set()  # 停用词集合
        if stopwords_file:
            with open(stopwords_file, encoding='utf-8') as f:
                self.stopwords = set(f.read().strip().splitlines())

    def prepare_data(self, data_dir):
        """
        加载和准备数据，包括文本读取、标签编码和数据集划分
        """
        texts = []
        labels = []

        # 遍历数据文件夹，读取文本和标签
        for folder in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder)
            if os.path.isdir(folder_path):
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)
                    with open(file_path, encoding='gbk', errors='ignore') as f:
                        texts.append(f.read())  # 添加文本数据
                        labels.append(folder)  # 标签为文件夹名称

        # 标签编码
        self.label_encoder.fit(labels)  # 根据标签训练编码器
        encoded_labels = self.label_encoder.transform(labels)  # 将标签转换为数字

        # 划分数据集为训练集、验证集和测试集
        X_train, X_temp, y_train, y_temp = train_test_split(
            texts, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        # 创建数据加载器
        self.train_loader = DataLoader(
            CommentDataset(X_train, y_train, self.tokenizer, self.stopwords, self.max_len),
            batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            CommentDataset(X_val, y_val, self.tokenizer, self.stopwords, self.max_len),
            batch_size=self.batch_size
        )
        self.test_loader = DataLoader(
            CommentDataset(X_test, y_test, self.tokenizer, self.stopwords, self.max_len),
            batch_size=self.batch_size
        )

        self.num_labels = len(self.label_encoder.classes_)  # 更新标签数量

        # 初始化模型
        self.model = BERTCNNClassifier(self.model_name, self.num_labels).to(self.device)

    def train_epoch(self, optimizer, scheduler, criterion):
        """
        训练一个epoch，包括前向传播、损失计算、反向传播和优化步骤
        """
        self.model.train()  # 设置模型为训练模式
        total_loss = 0
        correct = 0
        total = 0

        # 遍历训练数据
        for batch in tqdm(self.train_loader, desc="训练中"):
            optimizer.zero_grad()  # 清零梯度
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)  # 模型前向传播
            loss = criterion(outputs, labels)  # 计算损失
            total_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)  # 获取预测结果
            correct += torch.sum(preds == labels)  # 计算正确的样本数
            total += labels.size(0)  # 计算总样本数

            loss.backward()  # 反向传播
            optimizer.step()  # 更新优化器参数
            scheduler.step()  # 更新学习率

        # 返回每个epoch的损失和准确率
        return total_loss / len(self.train_loader), correct.double() / total

    def evaluate(self, data_loader, criterion):
        """
        评估模型的性能
        """
        self.model.eval()  # 设置模型为评估模式
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="评估中"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)  # 模型前向传播
                loss = criterion(outputs, labels)  # 计算损失
                total_loss += loss.item()

                _, preds = torch.max(outputs, dim=1)  # 获取预测结果
                correct += torch.sum(preds == labels)  # 计算正确的样本数
                total += labels.size(0)  # 计算总样本数

                all_preds.extend(preds.cpu().numpy())  # 保存预测结果
                all_labels.extend(labels.cpu().numpy())  # 保存实际标签

        # 返回评估结果
        return (
            total_loss / len(data_loader),
            correct.double() / total,
            all_preds,
            all_labels
        )

    def train(self, epochs=5, save_path='best_bert_cnn_model.pt'):
        """
        训练模型
        """
        criterion = nn.CrossEntropyLoss()  # 定义损失函数
        optimizer = AdamW(self.model.parameters(), lr=5e-5)  # 定义优化器
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_loader) * epochs
        )

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 10)

            # 训练一个epoch
            train_loss, train_acc = self.train_epoch(optimizer, scheduler, criterion)
            val_loss, val_acc, _, _ = self.evaluate(self.val_loader, criterion)

            print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
            print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")

            # 记录训练历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc.cpu().numpy())
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc.cpu().numpy())

            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(self.model.state_dict(), save_path)
                print(f"最佳模型已保存至 {save_path}")

    def plot_training_history(self):
        """
        绘制训练过程中的损失和准确率变化图
        """
        # 绘制损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], marker='o', label='train loss')
        plt.plot(self.history['val_loss'], marker='o', label='val loss')
        plt.title('Bert-CNN training and validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 绘制准确率曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_acc'], marker='o', label='train acc')
        plt.plot(self.history['val_acc'], marker='o', label='val acc')
        plt.title('Bert-CNN training and validation accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()

    def evaluate_and_plot(self, model_path='best_bert_cnn_model.pt'):
        """
        评估模型并绘制分类报告、混淆矩阵、ROC曲线
        """
        self.model.load_state_dict(torch.load(model_path))  # 加载最佳模型
        self.model.to(self.device)  # 将模型移动到GPU或CPU
        self.model.eval()  # 设置模型为评估模式

        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc, y_pred, y_true = self.evaluate(self.test_loader, criterion)  # 在测试集上评估
        print(f"\n测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.4f}")

        # 将预测结果和真实标签转回原始标签（文件夹名称）
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        y_true_labels = self.label_encoder.inverse_transform(y_true)

        # 输出分类报告
        print("\n分类报告:")
        print(classification_report(y_true_labels, y_pred_labels, digits=4, target_names=self.label_encoder.classes_))

        # 混淆矩阵
        cm = confusion_matrix(y_true_labels, y_pred_labels, labels=self.label_encoder.classes_)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('BERT-CNN Confusion Matrix')
        plt.show()

        # 计算ROC曲线
        probs = []
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs.append(torch.softmax(outputs, dim=1).cpu().numpy())

        test_probs = np.concatenate(probs)
        y_test_bin = label_binarize(y_true, classes=range(self.num_labels))

        # 绘制ROC曲线
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
        plt.title('Bert-CNN Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()


# 主程序
if __name__ == '__main__':
    data_dir = 'cn_news'  # 替换为你的数据集文件夹路径
    stopwords_file = '/root/autodl-tmp/test/baidu_stopwords.txt'  # 停用词文件路径

    classifier = BERTCNNClassifierWrapper(stopwords_file=stopwords_file)  # 初始化BERT-CNN分类器
    classifier.prepare_data(data_dir)  # 准备数据
    classifier.train(epochs=5, save_path='best_bert_cnn_model.pt')  # 训练模型并保存
    classifier.plot_training_history()  # 绘制训练历史
    classifier.evaluate_and_plot(model_path='best_bert_cnn_model.pt')  # 评估并绘制结果
