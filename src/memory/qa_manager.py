#控制从json中加载问答数据并向量化，包含检索功能
import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class QAManager:
    def __init__(self, data_path="assets/family_info.json", model_name="./assets/paraphrase-multilingual-MiniLM-L12-v2"):
        self.data_path = data_path
        self.model = SentenceTransformer(model_name)
        self.questions = []
        self.answers = []
        self.index = None
        self.dimension = 384  # MiniLM-L12-v2 embedding size

        # 初始化数据
        self.load_data()
        self.build_index()

    def load_data(self):
        """加载本地Q&A数据"""
        if os.path.exists(self.data_path):
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.questions = [q["question"] for q in data.get("questions", [])]
                self.answers = [q["answer"] for q in data.get("questions", [])]
                self.audio_file = [q["audio_file"] for q in data.get("questions", [])]
                self.command = [q["command"] for q in data.get("questions", [])]
        else:
            self.questions, self.answers = [], []

    def save_data(self):
        """保存到JSON"""
        data = {"questions": [{"question": q, "answer": a} for q, a in zip(self.questions, self.answers)]}
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        with open(self.data_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def build_index(self):
        """构建或重建向量索引"""
        if not self.questions:
            self.index = None
            return
        embeddings = self.model.encode(self.questions, convert_to_numpy=True, normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(self.dimension)  # 内积相似度（余弦）
        self.index.add(embeddings)

    def add_qa(self, question, answer):
        """新增Q&A并更新索引"""
        self.questions.append(question)
        self.answers.append(answer)
        self.save_data()
        # 增量更新向量
        emb = self.model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(emb)

    def query(self, text, top_k=1, threshold=0.5):
        """查询最相关的答案"""
        if self.index is None:
            return "知识库为空"
        emb = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        scores, indices = self.index.search(emb, top_k)
        best_score = scores[0][0]
        best_idx = indices[0][0]
        best_audio_file = self.audio_file[best_idx]
        best_command    = self.command[best_idx]
        if best_score >= threshold:
            return self.answers[best_idx], float(best_score),best_audio_file,best_command
        else:
            return "我暂时不知道怎么回答", float(best_score), None, "No Answer"
            # return "原神！启动！", float(best_score), None, None
