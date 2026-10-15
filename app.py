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

# Tùy chỉnh CSS giao diện
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

FEEDBACK_DATA_FILE = "student_feedbacks.csv"
TRAIN_DATA_FILE = "feedback_dataset.csv"

# MẬT KHẨU TRUY CẬP DÀNH CHO GIÁO VIÊN (Bạn có thể đổi mật khẩu tại đây)
TEACHER_PASSWORD = "thptdongphu2026"

# Danh mục khối và lớp học theo đúng thực tế
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
        df_train = pd.read_csv(TRAIN_DATA_FILE, encoding='utf-8-sig')
    except Exception:
        df_train = pd.read_csv(TRAIN_DATA_FILE, encoding='utf-8', errors='ignore')

    df_train['clean_text'] = df_train['text'].apply(vietnamese_preprocessing)
    
    model_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
        ('classifier', LinearSVC(random_state=42))
    ])
    model_pipeline.fit(df_train['clean_text'], df_train['sentiment'])
    return model_pipeline

model = train_sentiment_model()

# 4. Lưu phản hồi
def save_feedback(grade, classroom, subject, text, sentiment):
    new_entry = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Grade": grade,
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

# Banner tiêu đề trang
st.markdown("""
    <div class="main-header">
        <h1 style="margin:0; font-size: 28px;">🎓 Hệ Thống AI Phân Tích Cảm Xúc Phản Hồi Học Đường</h1>
        <p style="margin: 8px 0 0 0; opacity: 0.9; font-size: 14px;">
            Dự Án Nghiên Cứu Ứng Dụng Xử Lý Ngôn Ngữ Tự Nhiên & Học Máy Vào Giáo Dục THPT | Năm học 2026-2027
        </p>
    </div>
""", unsafe_allow_html=True)

menu = st.sidebar.radio("📌 Điều hướng hệ thống:", [
    "📝 Cổng Gửi Ý Kiến Ẩn Danh (Dành Cho Học Sinh)", 
    "📊 Bảng Điều Khiển & Phân Tích (Dành Cho Giáo Viên)"
])

# ================= TAB 1: CỔNG HỌC SINH =================
if menu == "📝 Cổng Gửi Ý Kiến Ẩn Danh (Dành Cho Học Sinh)":
    st.markdown("### 💌 Hộp thư đóng góp ý kiến sau tiết học")
    st.info("🔒 **Bảo mật tuyệt đối**: Danh tính học sinh không được lưu trữ. Mọi phản hồi chân thành của bạn sẽ giúp thầy cô lắng nghe, thấu hiểu và cải tiến phương pháp dạy học hiệu quả hơn!")
    
    with st.container():
        col_grade, col_class, col_sub = st.columns(3)
        with col_grade:
            selected_grade = st.selectbox("1. Chọn Khối:", list(CLASSES_BY_GRADE.keys()))
        with col_class:
            selected_class = st.selectbox("2. Chọn Lớp:", CLASSES_BY_GRADE[selected_grade])
        with col_sub:
            selected_subject = st.selectbox("3. Chọn Môn học / Hoạt động:", SUBJECTS_LIST)
            
        feedback_text = st.text_area(
            "4. Chia sẻ cảm nghĩ, khó khăn hoặc đề xuất của bạn về tiết học hôm nay:",
            placeholder="Ví dụ: Tiết học hôm nay thầy giảng rất dễ hiểu / Phần bài tập nhóm hơi nhanh em chưa kịp làm quen...",
            height=140
        )
        
        submit_btn = st.button("🚀 Gửi Phản Hồi Ngay", type="primary", use_container_width=True)
        
        if submit_btn:
            if not feedback_text.strip():
                st.warning("⚠️ Vui lòng nhập vài dòng cảm nghĩ trước khi nhấn gửi.")
            else:
                clean_input = vietnamese_preprocessing(feedback_text)
                prediction = model.predict([clean_input])[0]
                save_feedback(selected_grade, selected_class, selected_subject, feedback_text, prediction)
                st.success("🎉 Cảm ơn bạn! Ý kiến đóng góp ẩn danh đã được ghi nhận vào hệ thống thành công.")

# ================= TAB 2: DASHBOARD CHO GIÁO VIÊN (CÓ MẬT KHẨU) =================
else:
    st.markdown("### 📈 Báo Cáo & Phân Tích Dữ Liệu Cảm Xúc Học Sinh")
    
    # Quản lý trạng thái đăng nhập giáo viên trong phiên làm việc
    if "teacher_logged_in" not in st.session_state:
        st.session_state.teacher_logged_in = False

    if not st.session_state.teacher_logged_in:
        st.warning("🔒 Khu vực này dành riêng cho Ban Giám Hiệu và Thầy Cô giáo. Vui lòng nhập mã bảo mật để tiếp tục.")
        pwd_col1, pwd_col2 = st.columns([2, 1])
        with pwd_col1:
            entered_password = st.text_input("Nhập mã bảo vệ:", type="password", placeholder="Nhập mật khẩu giáo viên...")
        with pwd_col2:
            st.write("")
            st.write("")
            if st.button("Mở Khóa Bảng Điều Khiển", type="primary"):
                if entered_password == TEACHER_PASSWORD:
                    st.session_state.teacher_logged_in = True
                    st.rerun()
                else:
                    st.error("❌ Mật khẩu không chính xác. Vui lòng thử lại!")
    else:
        # Nút đăng xuất nhanh
        col_title, col_logout = st.columns([5, 1])
        with col_logout:
            if st.button("🚪 Đăng Xuất"):
                st.session_state.teacher_logged_in = False
                st.rerun()

        if not os.path.exists(FEEDBACK_DATA_FILE):
            st.warning("Chưa có dữ liệu phản hồi nào trong hệ thống.")
        else:
            try:
                df = pd.read_csv(FEEDBACK_DATA_FILE, encoding='utf-8-sig', on_bad_lines='skip')
            except Exception:
                df = pd.read_csv(FEEDBACK_DATA_FILE, encoding='utf-8', errors='ignore', on_bad_lines='skip')
            
            if "Grade" not in df.columns:
                df["Grade"] = df["Class"].apply(lambda x: f"Khối {str(x)[:2]}" if str(x)[:2].isdigit() else "Khối 10")
                
            # Bộ lọc tương tác
            st.markdown("##### 🔍 Bộ lọc dữ liệu:")
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                all_grades = ["Tất cả"] + sorted(list(df["Grade"].dropna().unique()))
                filter_grade = st.selectbox("Lọc theo Khối:", all_grades)
            with f_col2:
                if filter_grade != "Tất cả":
                    avail_classes = ["Tất cả"] + sorted(list(df[df["Grade"] == filter_grade]["Class"].dropna().unique()))
                else:
                    avail_classes = ["Tất cả"] + sorted(list(df["Class"].dropna().unique()))
                filter_class = st.selectbox("Lọc theo Lớp:", avail_classes)
            with f_col3:
                all_subjects = ["Tất cả"] + sorted(list(df["Subject"].dropna().unique()))
                filter_sub = st.selectbox("Lọc theo Môn học:", all_subjects)
                
            filtered_df = df.copy()
            if filter_grade != "Tất cả":
                filtered_df = filtered_df[filtered_df["Grade"] == filter_grade]
            if filter_class != "Tất cả":
                filtered_df = filtered_df[filtered_df["Class"] == filter_class]
            if filter_sub != "Tất cả":
                filtered_df = filtered_df[filtered_df["Subject"] == filter_sub]
                
            total = len(filtered_df)
            pos = len(filtered_df[filtered_df["Sentiment"] == "positive"])
            neu = len(filtered_df[filtered_df["Sentiment"] == "neutral"])
            neg = len(filtered_df[filtered_df["Sentiment"] == "negative"])
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Tổng lượt phản hồi", total)
            c2.metric("😊 Tích cực (Positive)", f"{pos} ({pos/total*100:.1f}%)" if total else "0")
            c3.metric("😐 Trung tính (Neutral)", f"{neu} ({neu/total*100:.1f}%)" if total else "0")
            c4.metric("🚨 Cần hỗ trợ (Negative)", f"{neg} ({neg/total*100:.1f}%)" if total else "0")
            
            st.markdown("---")
            
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                sentiment_map = {"positive": "Tích cực", "neutral": "Trung tính", "negative": "Cần hỗ trợ"}
                df_chart = filtered_df["Sentiment"].map(sentiment_map).value_counts().reset_index()
                df_chart.columns = ["Phân loại", "Số lượng"]
                fig_pie = px.pie(
                    df_chart, names="Phân loại", values="Số lượng", 
                    title="Tỷ lệ phân bố cảm xúc",
                    color="Phân loại",
                    color_discrete_map={"Tích cực": "#10B981", "Trung tính": "#F59E0B", "Cần hỗ trợ": "#EF4444"},
                    hole=0.4
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with chart_col2:
                st.markdown("##### 🚨 Phản hồi cần giáo viên lưu ý / hỗ trợ kịp thời:")
                negatives = filtered_df[filtered_df["Sentiment"] == "negative"][["Timestamp", "Class", "Subject", "Feedback"]]
                if len(negatives) > 0:
                    st.dataframe(negatives, use_container_width=True, hide_index=True)
                else:
                    st.success("Tuyệt vời! Không ghi nhận phản ánh khó khăn nào trong danh mục đã chọn.")

            st.markdown("---")
            st.markdown("##### 📋 Toàn bộ phản hồi ghi nhận:")
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)
            
            export_csv = filtered_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                "📥 Tải Báo Cáo Phân Tích (.CSV)", 
                data=export_csv, 
                file_name=f"Bao_Cao_Cam_Xuc_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", 
                mime="text/csv"
            )
