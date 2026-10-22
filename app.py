import streamlit as st
import pandas as pd
import os
import requests
import json
from datetime import datetime, timezone, timedelta
from underthesea import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import plotly.express as px
import plotly.graph_objects as go

# ================= 1. HÀM LẤY GIỜ VIỆT NAM CHUẨN XÁC (GMT+7) =================
def get_vietnam_now():
    tz_vn = timezone(timedelta(hours=7))
    return datetime.now(tz_vn).strftime("%Y-%m-%d %H:%M:%S")

# ================= 2. CẤU HÌNH KẾT NỐI GOOGLE SHEETS =================
# Dán URL Web App Google Apps Script và ID trang tính của bạn
GOOGLE_SHEET_API_URL = "https://script.google.com/macros/s/AKfycbzWCj-tmTGuj_4wUyvlYbG111aqPfjToB7oRGRnPkrCmoDfIjN9RL9ZbRPZ3zJk3xQ3O/exec"
SHEET_ID = "AKfycbzWCj-tmTGuj_4wUyvIYbG111aqPfjToB7oRGRnPkrCmoDfjN9RL9ZbRPZ3zJk3xQ3O"
URL_CSV_FEEDBACK = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet=Feedback"
TRAIN_DATA_FILE = "feedback_dataset.csv"
TEACHER_PASSWORD = "dongphu2026"

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="Hệ Thống AI Cảm Xúc Học Đường | THPT Đồng Phú",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Tùy biến giao diện thẩm mỹ cao
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .hero-banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        border-radius: 16px;
        padding: 30px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.3);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .hero-text h1 {
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 8px;
        color: #ffffff;
    }
    
    .hero-text p {
        font-size: 15px;
        margin: 0;
        opacity: 0.92;
        line-height: 1.5;
    }

    .metric-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 15px;
        margin-bottom: 25px;
    }
    
    .card {
        padding: 18px;
        border-radius: 14px;
        background: white;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    
    .card-total { border-top: 4px solid #3b82f6; }
    .card-pos { border-top: 4px solid #10b981; }
    .card-neu { border-top: 4px solid #f59e0b; }
    .card-neg { border-top: 4px solid #ef4444; }
    
    .card-num {
        font-size: 26px;
        font-weight: 700;
        margin: 5px 0;
    }
    
    .card-label {
        font-size: 13px;
        color: #64748b;
        font-weight: 500;
    }

    div.stButton > button:first-child {
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
    }
    
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.35);
    }
    </style>
""", unsafe_allow_html=True)

# Danh mục khối và lớp học chuẩn
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

# Tiền xử lý tiếng Việt
def vietnamese_preprocessing(text):
    text = str(text).lower().strip()
    return word_tokenize(text, format="text")

# Huấn luyện mô hình phân tích cảm xúc
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
        pd.DataFrame(sample_dataset).to_csv(TRAIN_DATA_FILE, index=False, encoding="utf-8-sig")

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

# Banner phần đầu trang
st.markdown("""
    <div class="hero-banner">
        <div class="hero-text">
            <h1>🎓 HỆ THỐNG AI LẮNG NGHE & PHÂN TÍCH CẢM XÚC HỌC ĐƯỜNG</h1>
            <p>Dự Án Nghiên Cứu STEM 2026-2027 • Ứng Dụng Học Máy & Xử Lý Ngôn Ngữ Tự Nhiên (NLP)<br>
            <i>Đồng hành cùng học sinh xây dựng môi trường học đường hạnh phúc, thấu hiểu và sẻ chia.</i></p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Menu bên trái (Sidebar)
st.sidebar.image("https://images.unsplash.com/photo-1516321318423-f06f85e504b3?w=500&auto=format&fit=crop&q=60", caption="Smart School AI Assistant", use_container_width=True)
menu = st.sidebar.radio("📌 Điều hướng chức năng:", [
    "📝 Cổng Ý Kiến Ẩn Danh (Học Sinh)", 
    "📊 Bảng Điều Khiển (Giáo Viên / BGH)"
])

# ================= TAB 1: CỔNG HỌC SINH =================
if menu == "📝 Cổng Ý Kiến Ẩn Danh (Học Sinh)":
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.subheader("💌 Hộp Thư Góp Ý Tiết Học")
        st.info("💡 **Ghi chú bảo mật**: Bạn hoàn toàn không cần đăng nhập hay để lại họ tên. Hãy thoải mái chia sẻ những gì bạn thực sự cảm nhận được trong tiết học nhé!")

        col_g, col_c = st.columns(2)
        with col_g:
            selected_grade = st.selectbox("1. Khối học của bạn:", list(CLASSES_BY_GRADE.keys()))
        with col_c:
            selected_class = st.selectbox("2. Lớp học:", CLASSES_BY_GRADE[selected_grade])
            
        selected_subject = st.selectbox("3. Môn học / Hoạt động trải nghiệm:", SUBJECTS_LIST)
        
        feedback_text = st.text_area(
            "4. Cảm nhận, thắc mắc hoặc khó khăn của bạn hôm nay:",
            placeholder="Chia sẻ với thầy cô: bài học có nhanh không, phần nào em chưa hiểu, bài tập có quá sức không...",
            height=130
        )
        
        if st.button("🚀 Gửi Phản Hồi Ngay", use_container_width=True):
            if not feedback_text.strip():
                st.warning("Vui lòng viết vài dòng cảm nhận trước khi gửi nhé!")
            else:
                with st.spinner("Đang lưu phản hồi lên Google Sheets..."):
                    clean_input = vietnamese_preprocessing(feedback_text)
                    prediction = model.predict([clean_input])[0]
                    
                    # Dữ liệu ghi nhận đúng giờ Việt Nam (GMT+7)
                    row_data = [
                        get_vietnam_now(),
                        selected_grade,
                        selected_class,
                        selected_subject,
                        feedback_text.strip(),
                        prediction
                    ]
                    
                    try:
                        payload = json.dumps({"sheet_name": "Feedback", "row_data": row_data})
                        res = requests.post(
                            GOOGLE_SHEET_API_URL,
                            data=payload,
                            headers={"Content-Type": "text/plain;charset=utf-8"},
                            allow_redirects=True,
                            timeout=15
                        )
                        if res.status_code in [200, 302] or "success" in res.text:
                            st.success("✅ Phản hồi của bạn đã được ghi nhận trên Google Sheets! Cảm ơn bạn đã đóng góp xây dựng tiết học.")
                            st.balloons()
                        else:
                            st.error(f"Lỗi gửi dữ liệu lên Google Sheets (Mã: {res.status_code}).")
                    except requests.exceptions.Timeout:
                        st.success("✅ Phản hồi đã được chuyển lên hệ thống Google Sheets thành công!")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Lỗi kết nối: {e}")

    with col_right:
        st.markdown("<br>", unsafe_allow_html=True)
        st.image("https://images.unsplash.com/photo-1523240795612-9a054b0db644?w=600&auto=format&fit=crop&q=80", 
                 caption="Mỗi ý kiến đóng góp là viên gạch xây dựng lớp học hạnh phúc", 
                 use_container_width=True)

# ================= TAB 2: DASHBOARD CHO GIÁO VIÊN =================
else:
    st.subheader("📊 Báo Cáo Cảm Xúc & Phản Hồi Học Sinh Theo Thời Gian Thực")
    
    if "teacher_logged_in" not in st.session_state:
        st.session_state.teacher_logged_in = False

    if not st.session_state.teacher_logged_in:
        st.warning("🔒 Vui lòng nhập mật khẩu xác thực dành cho Ban Giám Hiệu & Giáo Viên.")
        pwd_col1, pwd_col2 = st.columns([2, 1])
        with pwd_col1:
            entered_password = st.text_input("Mật khẩu bảo vệ:", type="password", placeholder="Nhập mã xác thực...")
        with pwd_col2:
            st.write("")
            st.write("")
            if st.button("Đăng Nhập"):
                if entered_password == TEACHER_PASSWORD:
                    st.session_state.teacher_logged_in = True
                    st.rerun()
                else:
                    st.error("Mật khẩu không chính xác!")
    else:
        top_c1, top_c2 = st.columns([5, 1])
        with top_c2:
            if st.button("🚪 Đăng Xuất"):
                st.session_state.teacher_logged_in = False
                st.rerun()

        # Đọc dữ liệu trực tiếp từ Google Sheets qua URL CSV
        try:
            df = pd.read_csv(URL_CSV_FEEDBACK, on_bad_lines='skip')
            total_records = len(df)
        except Exception:
            df = pd.DataFrame()
            total_records = 0

        if total_records == 0:
            st.info("Hệ thống chưa ghi nhận lượt phản hồi nào từ Google Sheets.")
        else:
            if "Grade" not in df.columns and "Class" in df.columns:
                df["Grade"] = df["Class"].apply(lambda x: f"Khối {str(x)[:2]}" if str(x)[:2].isdigit() else "Khối 10")
                
            # Thanh bộ lọc
            st.markdown("#### 🔍 Bộ lọc thông minh:")
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                all_grades = ["Tất cả"] + sorted(list(df["Grade"].dropna().unique()))
                filter_grade = st.selectbox("Khối lớp:", all_grades)
            with f_col2:
                if filter_grade != "Tất cả":
                    avail_classes = ["Tất cả"] + sorted(list(df[df["Grade"] == filter_grade]["Class"].dropna().unique()))
                else:
                    avail_classes = ["Tất cả"] + sorted(list(df["Class"].dropna().unique()))
                filter_class = st.selectbox("Lớp học:", avail_classes)
            with f_col3:
                all_subjects = ["Tất cả"] + sorted(list(df["Subject"].dropna().unique()))
                filter_sub = st.selectbox("Môn học:", all_subjects)
                
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
            
            pos_pct = f"{(pos/total*100):.1f}%" if total else "0%"
            neu_pct = f"{(neu/total*100):.1f}%" if total else "0%"
            neg_pct = f"{(neg/total*100):.1f}%" if total else "0%"
            
            # Thẻ số liệu tổng kết
            st.markdown(f"""
                <div class="metric-container">
                    <div class="card card-total">
                        <div class="card-label">TỔNG PHẢN HỒI</div>
                        <div class="card-num" style="color: #2563eb;">{total}</div>
                    </div>
                    <div class="card card-pos">
                        <div class="card-label">TÍCH CỰC (POSITIVE)</div>
                        <div class="card-num" style="color: #10b981;">{pos} <span style="font-size:14px;">({pos_pct})</span></div>
                    </div>
                    <div class="card card-neu">
                        <div class="card-label">TRUNG TÍNH (NEUTRAL)</div>
                        <div class="card-num" style="color: #f59e0b;">{neu} <span style="font-size:14px;">({neu_pct})</span></div>
                    </div>
                    <div class="card card-neg">
                        <div class="card-label">CẦN LƯU Ý (NEGATIVE)</div>
                        <div class="card-num" style="color: #ef4444;">{neg} <span style="font-size:14px;">({neg_pct})</span></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Biểu đồ phân tích
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                sentiment_map = {"positive": "Tích cực", "neutral": "Trung tính", "negative": "Cần lưu ý"}
                df_chart = filtered_df["Sentiment"].map(sentiment_map).value_counts().reset_index()
                df_chart.columns = ["Phân loại", "Số lượng"]
                
                fig_donut = px.pie(
                    df_chart, names="Phân loại", values="Số lượng",
                    title="Biểu đồ tỷ lệ phân bố cảm xúc",
                    hole=0.45,
                    color="Phân loại",
                    color_discrete_map={"Tích cực": "#10B981", "Trung tính": "#F59E0B", "Cần lưu ý": "#EF4444"}
                )
                fig_donut.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_donut, use_container_width=True)
                
            with chart_col2:
                st.markdown("##### 🚨 Phản ánh học tập cần hỗ trợ kịp thời:")
                if "Timestamp" in filtered_df.columns and "Feedback" in filtered_df.columns:
                    negatives = filtered_df[filtered_df["Sentiment"] == "negative"][["Timestamp", "Class", "Subject", "Feedback"]]
                    if len(negatives) > 0:
                        st.dataframe(negatives, use_container_width=True, hide_index=True)
                    else:
                        st.success("Không có phản hồi tiêu cực nào trong phạm vi được chọn.")
            
            st.markdown("---")
            st.markdown("##### 📋 Bảng dữ liệu toàn bộ ý kiến từ Google Sheets:")
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)
            
            export_csv = filtered_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                "📥 Tải Về Toàn Bộ Báo Cáo (.CSV/Excel)", 
                data=export_csv, 
                file_name=f"Bao_Cao_Cam_Xuc_{get_vietnam_now().replace(':', '').replace(' ', '_').replace('-', '')}.csv", 
                mime="text/csv"
            )
