import os
import re
from colorama import Fore
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
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
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 防止负号显示问题


# 定义数据集类，继承自Dataset
class CommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, stopwords, max_len=128):
        """
        数据集初始化，包括文本数据、标签、tokenizer、停用词表以及最大长度设置
        """
        self.tokenizer = tokenizer  # 使用的BERT tokenizer
        self.texts = texts  # 输入文本
        self.labels = labels  # 输入标签
        self.stopwords = stopwords  # 停用词
        self.max_len = max_len  # 最大长度，超过会截断，短于会填充

    def __len__(self):
        """
        返回数据集的大小
        """
        return len(self.texts)

    def preprocess_text(self, text):
        """
        对文本进行预处理，包括去除停用词和特殊字符
        """
        text = re.sub(r'[^A-Za-z0-9\u4e00-\u9fa5]+', ' ', text)  # 只保留中文字符和英文、数字
        text = text.strip()  # 去除两端空格
        words = text.split()  # 按空格分词
        words = [word for word in words if word not in self.stopwords]  # 去除停用词
        return ' '.join(words)  # 重新组合文本

    def __getitem__(self, idx):
        """
        获取数据集中的一个样本，返回处理后的input_ids、attention_mask和label
        """
        text = self.texts[idx]
        text = self.preprocess_text(text)  # 预处理文本
        encoding = self.tokenizer.encode_plus(
            str(text),
            max_length=self.max_len,  # 截断或填充至最大长度
            padding='max_length',  # 填充
            truncation=True,  # 截断
            return_attention_mask=True,  # 返回attention_mask
            return_tensors='pt'  # 返回Pytorch tensor格式
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx] if self.labels is not None else -1, dtype=torch.long)
        }


# 定义BERT分类器类
class BERTClassifier:
    def __init__(self, model_name='bert-base-chinese', num_labels=None, batch_size=32, max_len=128,
                 stopwords_file=None):
        """
        初始化BERT分类器，加载预训练模型和tokenizer
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 使用GPU或者CPU
        self.tokenizer = BertTokenizer.from_pretrained(model_name)  # 加载tokenizer
        self.batch_size = batch_size  # 批量大小
        self.max_len = max_len  # 最大长度
        self.model_name = model_name  # 模型名称
        self.num_labels = num_labels  # 标签数
        self.label_encoder = LabelEncoder()  # 标签编码器，用于将标签转换为数字
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}  # 训练过程中的历史记录
        self.best_val_acc = 0  # 保存最佳验证准确率

        self.stopwords = set()  # 停用词集合
        if stopwords_file:
            with open(stopwords_file, encoding='utf-8') as f:
                self.stopwords = set(f.read().strip().splitlines())  # 从文件加载停用词

    def prepare_data(self, data_dir):
        """
        准备数据集，读取文本数据并划分训练集、验证集和测试集
        """
        texts = []
        labels = []

        # 使用文件夹名称作为标签
        for folder in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder)
            if os.path.isdir(folder_path):
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)
                    with open(file_path, encoding='gbk', errors='ignore') as f:
                        texts.append(f.read())  # 读取文件内容
                        labels.append(folder)  # 文件夹名称作为标签

        # 标签编码
        self.label_encoder.fit(labels)
        encoded_labels = self.label_encoder.transform(labels)

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

        self.num_labels = len(self.label_encoder.classes_)  # 标签类别数

        # 初始化BERT模型
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        ).to(self.device)

    def train_epoch(self, optimizer, scheduler):
        """
        训练一个epoch
        """
        self.model.train()  # 设置模型为训练模式
        total_loss = 0
        correct = 0
        total = 0

        # 遍历训练数据加载器
        for batch in tqdm(self.train_loader, desc="训练中"):
            optimizer.zero_grad()  # 梯度清零
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # 前向传播
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            # 预测
            _, preds = torch.max(outputs.logits, dim=1)
            correct += torch.sum(preds == labels)
            total += labels.size(0)

            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            scheduler.step()  # 更新学习率

        # 返回损失和准确率
        return total_loss / len(self.train_loader), correct.double() / total

    def evaluate(self, data_loader):
        """
        在验证集或测试集上进行评估
        """
        self.model.eval()  # 设置模型为评估模式
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        # 不计算梯度
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="评估中"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += outputs.loss.item()

                # 预测
                _, preds = torch.max(outputs.logits, dim=1)
                correct += torch.sum(preds == labels)
                total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 返回评估结果
        return (
            total_loss / len(data_loader),
            correct.double() / total,
            all_preds,
            all_labels
        )

    def train(self, epochs=5, save_path='best_bert_model.pt'):
        """
        训练模型
        """
        optimizer = AdamW(self.model.parameters(), lr=3e-5)  # 使用AdamW优化器
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_loader) * epochs
        )

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 10)

            train_loss, train_acc = self.train_epoch(optimizer, scheduler)  # 训练一个epoch
            val_loss, val_acc, _, _ = self.evaluate(self.val_loader)  # 在验证集上评估

            print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
            print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")

            # 记录训练过程
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
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], marker='o', label='train loss')
        plt.plot(self.history['val_loss'], marker='o', label='val loss')
        plt.title('Bert training and validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_acc'], marker='o', label='train acc')
        plt.plot(self.history['val_acc'], marker='o', label='val acc')
        plt.title('Bert training and validation accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()

    def evaluate_and_plot(self, model_path='best_bert_model.pt'):
        """
        评估测试集并绘制相关图表
        """
        self.model.load_state_dict(torch.load(model_path))  # 加载保存的最佳模型
        self.model.to(self.device)
        self.model.eval()

        test_loss, test_acc, y_pred, y_true = self.evaluate(self.test_loader)  # 在测试集上评估
        print(f"\n测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.4f}")

        # 转换回原始标签（文件夹名称）
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        y_true_labels = self.label_encoder.inverse_transform(y_true)

        # 打印分类报告
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
        plt.title('BERT Confusion Matrix')
        plt.show()

        # 绘制ROC曲线
        probs = []
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs.append(torch.softmax(outputs.logits, dim=1).cpu().numpy())

        test_probs = np.concatenate(probs)
        y_test_bin = label_binarize(y_true, classes=range(self.num_labels))

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
        plt.title('Bert Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    data_dir = 'cn_news'  # 数据目录
    stopwords_file = '/root/autodl-tmp/test/baidu_stopwords.txt'  # 停用词文件
    classifier = BERTClassifier(stopwords_file=stopwords_file)  # 初始化BERT分类器
    classifier.prepare_data(data_dir)  # 准备数据
    classifier.train(epochs=5, save_path='best_bert_model.pt')  # 训练模型
    classifier.plot_training_history()  # 绘制训练历史图
    classifier.evaluate_and_plot(model_path='best_bert_model.pt')  # 评估测试集并绘图
