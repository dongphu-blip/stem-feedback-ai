import streamlit as st
import pandas as pd
import os
from datetime import datetime
from underthesea import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import plotly.express as px

# 1. Cấu hình giao diện trang
st.set_page_config(
    page_title="AI Classroom Sentiment Analyzer",
    page_icon="🎓",
    layout="wide"
)

# Tùy chỉnh CSS giao diện hiện đại, chuyên nghiệp
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

FEEDBACK_DATA_FILE = "student_feedbacks.csv"
TRAIN_DATA_FILE = "feedback_dataset.csv"

# Danh mục khối và lớp học chuẩn của trường
CLASSES_BY_GRADE = {
    "Khối 10": [f"10TN{i}" for i in range(1, 6)] + [f"10XH{i}" for i in range(1, 6)],
    "Khối 11": [f"11TN{i}" for i in range(1, 5)] + [f"11XH{i}" for i in range(1, 8)],
    "Khối 12": [f"12TN{i}" for i in range(1, 7)] + [f"12XH{i}" for i in range(1, 7)]
}

SUBJECTS_LIST = [
    "Toán học", "Ngữ văn", "Tiếng Anh", "Tin học", 
    "Vật lý", "Hóa học", "Sinh học", "Lịch sử", 
    "Địa lý", "GDCD / GDKT&PL", "Hoạt động trải nghiệm / Hướng nghiệp / SHL"
]

# 2. Tiền xử lý văn bản tiếng Việt
def vietnamese_preprocessing(text):
    text = str(text).lower().strip()
    return word_tokenize(text, format="text")

# 3. Huấn luyện mô hình AI
@st.cache_resource
def train_sentiment_model():
    if not os.path.exists(TRAIN_DATA_FILE):
        sample_dataset = {
            "text": [
                "Thầy giảng bài hôm nay rất hay và dễ hiểu",
                "Tiết học thực hành rất hào hứng và thú vị",
                "Em rất thích bài thuyết trình hôm nay",
                "Không khí lớp học hôm nay rất tích cực",
                "Em đã hiểu bài ngay tại lớp cảm ơn cô",
                "Nội dung bài học bình thường không có gì đặc biệt",
                "Bài học hôm nay vừa sức",
                "Em ghi chép bài đầy đủ rồi",
                "Tốc độ giảng dạy bình thường",
                "Tiết học diễn ra như mọi ngày",
                "Bài tập về nhà môn này quá nhiều làm em bị quá tải",
                "Thầy nói hơi nhanh em không theo kịp bài giảng",
                "Em cảm thấy rất áp lực và mệt mỏi trong giờ học",
                "Phòng máy tính chập chờn không kết nối được mạng",
                "Kiến thức phần này quá khó hiểu và trừu tượng",
                "Bài tập khó, chưa nhớ hết kiến thức"
            ],
            "sentiment": [
                "positive", "positive", "positive", "positive", "positive",
                "neutral", "neutral", "neutral", "neutral", "neutral",
                "negative", "negative", "negative", "negative", "negative", "negative"
            ]
        }
        df_init = pd.DataFrame(sample_dataset)
        df_init.to_csv(TRAIN_DATA_FILE, index=False, encoding="utf-8-sig")

    try:
        df_train = pd.read_csv(TRA