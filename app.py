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

FEEDBACK_DATA_FILE = "student_feedbacks.csv"
TRAIN_DATA_FILE = "feedback_dataset.csv"

# 2. Tiền xử lý văn bản tiếng Việt
def vietnamese_preprocessing(text):
    text = str(text).lower().strip()
    tokenized_text = word_tokenize(text, format="text")
    return tokenized_text

# 3. Huấn luyện mô hình AI
@st.cache_resource
def train_sentiment_model():
    # Tự động tạo tệp dữ liệu mẫu chuẩn UTF-8-SIG nếu chưa có
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
                "Kiến thức phần này quá khó hiểu và trừu tượng"
            ],
            "sentiment": [
                "positive", "positive", "positive", "positive", "positive",
                "neutral", "neutral", "neutral", "neutral", "neutral",
                "negative", "negative", "negative", "negative", "negative"
            ]
        }
        df_init = pd.DataFrame(sample_dataset)
        df_init.to_csv(TRAIN_DATA_FILE, index=False, encoding="utf-8-sig")

    # Đọc dữ liệu huấn luyện an toàn với nhiều phương án mã hóa
    try:
        df_train = pd.read_csv(TRAIN_DATA_FILE, encoding='utf-8-sig')
    except Exception:
        try:
            df_train = pd.read_csv(TRAIN_DATA_FILE, encoding='utf-8')
        except Exception:
            df_train = pd.read_csv(TRAIN_DATA_FILE, encoding='cp1258')

    df_train['clean_text'] = df_train['text'].apply(vietnamese_preprocessing)
    
    model_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
        ('classifier', LinearSVC(random_state=42))
    ])
    model_pipeline.fit(df_train['clean_text'], df_train['sentiment'])
    return model_pipeline

model = train_sentiment_model()

# 4. Lưu phản hồi học sinh
def save_feedback(classroom, subject, text, sentiment):
    new_entry = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Class": classroom,
        "Subject": subject,
        "Feedback": text,
        "Sentiment": sentiment
    }
    df_new = pd.DataFrame([new_entry])
    if not os.path.exists(FEEDBACK_DATA_FILE):
        df_new.to_csv(FEEDBACK_DATA_FILE, index=False, encoding="utf-8-sig")
    else:
        df_new.to_csv(FEEDBACK_DATA_FILE, mode='a', header=False, index=False, encoding="utf-8-sig")

# 5. Giao diện người dùng
st.title("🎓 Hệ Thống AI Phân Tích Cảm Xúc Phản Hồi Học Đường")
st.caption("AI-Based Sentiment Analysis Tool for Classroom Student Feedback | STEM 2026-2027")

menu = st.sidebar.radio("Chọn chức năng / Navigation:", ["Gửi Phản Hồi Ẩn Danh (Student Portal)", "Bảng Điều Khiển Giáo Viên (Teacher Dashboard)"])

# TAB 1: CỔNG HỌC SINH
if menu == "Gửi Phản Hồi Ẩn Danh (Student Portal)":
    st.subheader("📝 Hộp Thư Đóng Góp Ý Kiến Ẩn Danh")
    st.info("Hệ thống hoàn toàn ẩn danh. Ý kiến của bạn sẽ giúp thầy cô điều chỉnh phương pháp dạy học phù hợp và xây dựng lớp học hạnh phúc hơn!")
    
    col1, col2 = st.columns(2)
    with col1:
        grade_class = st.selectbox("Chọn Lớp:", ["10TN5", "Khối 10", "Khối 11", "Khối 12"])
    with col2:
        subject = st.selectbox("Môn học / Hoạt động:", ["Tin học", "Toán", "Vật lý", "Hóa học", "Sinh học", "Hoạt động trải nghiệm / Sinh hoạt lớp"])
        
    feedback_text = st.text_area("Nhập suy nghĩ, cảm xúc hoặc khó khăn của bạn trong tiết học hôm nay:", height=130)
    
    if st.button("Gửi Phản Hồi Ngay", type="primary"):
        if feedback_text.strip() == "":
            st.warning("Vui lòng nhập nội dung ý kiến trước khi gửi.")
        else:
            clean_input = vietnamese_preprocessing(feedback_text)
            prediction = model.predict([clean_input])[0]
            save_feedback(grade_class, subject, feedback_text, prediction)
            st.success("✅ Cảm ơn bạn! Ý kiến của bạn đã được ghi nhận ẩn danh thành công.")

# TAB 2: DASHBOARD CHO GIÁO VIÊN
else:
    st.subheader("📊 Báo Cáo & Phân Tích Cảm Xúc Lớp Học Theo Thời Gian Thực")
    
    if not os.path.exists(FEEDBACK_DATA_FILE):
        st.warning("Hiện chưa có dữ liệu phản hồi nào được ghi nhận.")
    else:
        df = pd.read_csv(FEEDBACK_DATA_FILE, encoding='utf-8-sig')
        selected_class = st.selectbox("Lọc theo lớp:", ["Tất cả"] + list(df["Class"].unique()))
        if selected_class != "Tất cả":
            df = df[df["Class"] == selected_class]
            
        total_feedback = len(df)
        pos_count = len(df[df["Sentiment"] == "positive"])
        neu_count = len(df[df["Sentiment"] == "neutral"])
        neg_count = len(df[df["Sentiment"] == "negative"])
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tổng phản hồi", total_feedback)
        c1_ratio = f"{(pos_count/total_feedback)*100:.1f}%" if total_feedback else "0%"
        c2.metric("Tích cực (Positive)", f"{pos_count} ({c1_ratio})")
        c3.metric("Trung tính (Neutral)", neu_count)
        c4.metric("Cần lưu ý / Tiêu cực (Negative)", neg_count)
        
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            sentiment_map = {"positive": "Tích cực", "neutral": "Trung tính", "negative": "Tiêu cực"}
            df_plot = df["Sentiment"].map(sentiment_map).value_counts().reset_index()
            df_plot.columns = ["Cảm xúc", "Số lượng"]
            fig_pie = px.pie(df_plot, names="Cảm xúc", values="Số lượng", title="Tỷ lệ cảm xúc chung của học sinh",
                             color="Cảm xúc",
                             color_discrete_map={"Tích cực": "#28a745", "Trung tính": "#ffc107", "Tiêu cực": "#dc3545"})
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_chart2:
            st.markdown("##### 🚨 Phản Hồi Cần Quan Tâm Đặc Biệt (Negative Feedback):")
            neg_df = df[df["Sentiment"] == "negative"][["Timestamp", "Subject", "Feedback"]]
            if len(neg_df) > 0:
                st.dataframe(neg_df, use_container_width=True, hide_index=True)
            else:
                st.success("Không có phản hồi tiêu cực nào ghi nhận!")
                
        st.markdown("---")
        st.markdown("##### Danh sách toàn bộ phản hồi thu nhận:")
        st.dataframe(df, use_container_width=True)
        
        csv_data = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("📥 Xuất Báo Cáo Dữ Liệu (.CSV)", data=csv_data, file_name="Bao_Cao_Cam_Xuc_Hoc_Sinh.csv", mime="text/csv")