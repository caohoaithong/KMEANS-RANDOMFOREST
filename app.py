
# =====================================================
# 1. IMPORT THU VIEN
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    confusion_matrix,
    classification_report
)


# =====================================================
# 2. CAU HINH TRANG
# =====================================================

st.set_page_config(
    page_title="Phân cụm và dự đoán rượu vang",
    page_icon="🍷",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#e0f2fe,#f5f3ff,#ffffff);
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
}

h1 {
    color: #123b70 !important;
    font-weight: 800 !important;
}

h2, h3 {
    color: #174a8b !important;
}

div[data-testid="stMetric"] {
    background: white;
    border-radius: 15px;
    padding: 18px;
    box-shadow: 0 5px 15px rgba(30,64,175,0.12);
}

.section {
    background: linear-gradient(120deg,#ffffff,#dbeafe);
    padding: 15px 20px;
    border-left: 6px solid #2563eb;
    border-radius: 12px;
    margin: 20px 0;
    font-size: 24px;
    font-weight: 800;
    color: #123b70;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. KHAI BAO THUOC TINH
# =====================================================

FEATURES = [
    "fixed acidity",
    "total sulfur dioxide",
    "pH",
    "alcohol"
]

TARGET = "quality"

st.markdown("""
# 🍷 PHÂN CỤM VÀ DỰ ĐOÁN CHẤT LƯỢNG RƯỢU VANG

Hệ thống ứng dụng hai thuật toán Machine Learning:

- K-means: Phân cụm dữ liệu.
- Random Forest: Dự đoán chất lượng rượu vang.
""")

st.divider()


# =====================================================
# 4. DOC FILE EXCEL VA KIEM TRA TEN COT
# =====================================================

st.markdown(
    '<div class="section">📂 1. TRUY XUẤT DỮ LIỆU</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Chọn file Excel dữ liệu rượu vang",
    type=["xlsx", "xls"]
)

def normalize_name(value):
    return " ".join(str(value).strip().lower().split())

if uploaded_file is None:
    st.info("Vui lòng tải file Excel để bắt đầu.")
    st.stop()

try:
    # Đọc thô, không giả định dòng tiêu đề
    raw = pd.read_excel(uploaded_file, header=None)

    required = set(
        normalize_name(c)
        for c in FEATURES + [TARGET]
    )

    header_index = None

    # Tìm dòng chứa đầy đủ các tên cột cần thiết
    for i in range(min(15, len(raw))):
        row_names = set(
            normalize_name(v)
            for v in raw.iloc[i].dropna()
        )

        if required.issubset(row_names):
            header_index = i
            break

    if header_index is None:
        st.error("Không tìm thấy dòng tiêu đề chứa đủ các cột yêu cầu.")
        st.write("Dữ liệu thô để kiểm tra:")
        st.dataframe(raw.head(15))
        st.stop()

    # Đọc lại với dòng tiêu đề chính xác
    uploaded_file.seek(0)

    df = pd.read_excel(
        uploaded_file,
        header=header_index
    )

    # Chuẩn hóa tên cột
    df.columns = [
        normalize_name(c)
        for c in df.columns
    ]

    # Đưa tên cột về đúng chuẩn
    df = df.rename(columns={
        "fixed acidity": "fixed acidity",
        "total sulfur dioxide": "total sulfur dioxide",
        "ph": "pH",
        "alcohol": "alcohol",
        "quality": "quality"
    })

    # Kiểm tra cột
    missing_cols = [
        c for c in FEATURES + [TARGET]
        if c not in df.columns
    ]

    if missing_cols:
        st.error(f"Thiếu các cột: {missing_cols}")
        st.write("Các cột hiện có:", df.columns.tolist())
        st.stop()

    st.success("Đọc file Excel thành công!")

except Exception as e:
    st.error(f"Lỗi đọc file: {e}")
    st.stop()


# =====================================================
# 5. KIEM TRA DU LIEU
# =====================================================

st.markdown(
    '<div class="section">📋 2. KIỂM TRA DỮ LIỆU</div>',
    unsafe_allow_html=True
)

df = df.copy()

# Chuyển các cột cần thiết sang dạng số
for col in FEATURES + [TARGET]:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# Loại bỏ giá trị vô hạn
df[FEATURES + [TARGET]] = df[
    FEATURES + [TARGET]
].replace([np.inf, -np.inf], np.nan)

# Chỉ giữ các dòng có nhãn quality hợp lệ
df = df.dropna(subset=[TARGET])

# Quality phải là số nguyên để phân loại
df = df[
    df[TARGET] % 1 == 0
].copy()

if len(df) < 4:
    st.error("Không đủ dữ liệu hợp lệ để xây dựng mô hình.")
    st.stop()

# Loại bỏ cột STT khỏi dữ liệu huấn luyện
X = df[FEATURES].copy()
y = df[TARGET].astype(int).copy()

c1, c2, c3 = st.columns(3)

c1.metric("Số lượng mẫu", len(df))
c2.metric("Số thuộc tính", len(FEATURES))
c3.metric("Số loại quality", y.nunique())

st.write("### 10 dòng dữ liệu đầu tiên")
st.dataframe(df.head(10), use_container_width=True)

st.write("### Kiểm tra dữ liệu thiếu")
st.dataframe(
    df[FEATURES + [TARGET]].isnull().sum().rename(
        "Số giá trị thiếu"
    )
)

st.write("### Thống kê dữ liệu")
st.dataframe(df[FEATURES + [TARGET]].describe())


# =====================================================
# 6. XU LY VA CHUAN HOA DU LIEU
# =====================================================

st.markdown(
    '<div class="section">⚙️ 3. TIỀN XỬ LÝ DỮ LIỆU</div>',
    unsafe_allow_html=True
)

# K-means: xử lý thiếu và chuẩn hóa
imputer = SimpleImputer(strategy="median")

X_imputed = pd.DataFrame(
    imputer.fit_transform(X),
    columns=FEATURES,
    index=X.index
)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_imputed)

st.success("Đã xử lý và chuẩn hóa dữ liệu.")

st.write("### Dữ liệu sau chuẩn hóa")
st.dataframe(
    pd.DataFrame(
        X_scaled,
        columns=FEATURES,
        index=X.index
    ).head(10)
)


# =====================================================
# 7. ELBOW METHOD
# =====================================================

st.markdown(
    '<div class="section">📈 4. ELBOW METHOD</div>',
    unsafe_allow_html=True
)

max_k = min(20, len(X_scaled) - 1)

inertias = []

for k in range(1, max_k + 1):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
        max_iter=300
    )

    model.fit(X_scaled)

    inertias.append(model.inertia_)

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    range(1, max_k + 1),
    inertias,
    marker="o",
    linewidth=2
)

ax.set_title("ELBOW METHOD - K = 1 ĐẾN 20")
ax.set_xlabel("Số cụm K")
ax.set_ylabel("Inertia (WCSS)")
ax.set_xticks(range(1, max_k + 1))
ax.grid(alpha=0.3)

st.pyplot(fig)
plt.close(fig)

# Cho phép lựa chọn K
k_selected = st.slider(
    "Chọn số cụm K để phân tích",
    min_value=2,
    max_value=max_k,
    value=min(3, max_k)
)

st.info(
    f"Số cụm đang được lựa chọn: K = {k_selected}. "
    "Hãy đối chiếu với biểu đồ Elbow trước khi quyết định."
)


# =====================================================
# 8. HUAN LUYEN K-MEANS
# =====================================================

st.markdown(
    '<div class="section">🔵 5. PHÂN CỤM K-MEANS</div>',
    unsafe_allow_html=True
)

kmeans = KMeans(
    n_clusters=k_selected,
    random_state=42,
    n_init=10,
    max_iter=300
)

clusters = kmeans.fit_predict(X_scaled)

df_result = df.copy()
df_result["Cluster"] = clusters

c1, c2, c3 = st.columns(3)

c1.metric("Số cụm", k_selected)
c2.metric("Số vòng lặp", kmeans.n_iter_)
c3.metric("Inertia", f"{kmeans.inertia_:.4f}")

st.write("### Kết quả phân cụm")
st.dataframe(df_result, use_container_width=True)

st.write("### Số lượng mẫu trong từng cụm")

cluster_count = df_result["Cluster"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(9, 4))

sns.barplot(
    x=cluster_count.index,
    y=cluster_count.values,
    ax=ax
)

ax.set_xlabel("Cluster")
ax.set_ylabel("Số lượng mẫu")
ax.set_title("SỐ LƯỢNG MẪU TRONG TỪNG CỤM")

st.pyplot(fig)
plt.close(fig)


# =====================================================
# 9. PHAN TICH Y NGHIA TUNG CUM
# =====================================================

st.markdown(
    '<div class="section">📊 6. PHÂN TÍCH ĐẶC ĐIỂM CỤM</div>',
    unsafe_allow_html=True
)

cluster_mean = df_result.groupby("Cluster")[
    FEATURES + [TARGET]
].mean()

st.write("### Giá trị trung bình của từng cụm")
st.dataframe(cluster_mean)

cluster_ratio = (
    cluster_count / len(df_result) * 100
).rename("Tỷ lệ (%)")

st.write("### Tỷ lệ mẫu từng cụm")
st.dataframe(cluster_ratio)

for cluster_id in sorted(df_result["Cluster"].unique()):

    values = cluster_mean.loc[cluster_id]

    st.markdown(f"#### 🔹 Cụm {cluster_id}")

    st.write(
        f"- Số lượng mẫu: {cluster_count[cluster_id]}"
    )

    st.write(
        f"- Tỷ lệ: {cluster_ratio[cluster_id]:.2f}%"
    )

    st.write(
        f"- Fixed acidity trung bình: {values['fixed acidity']:.2f}"
    )

    st.write(
        f"- Total sulfur dioxide trung bình: "
        f"{values['total sulfur dioxide']:.2f}"
    )

    st.write(
        f"- pH trung bình: {values['pH']:.2f}"
    )

    st.write(
        f"- Alcohol trung bình: {values['alcohol']:.2f}"
    )

    st.write(
        f"- Quality trung bình: {values['quality']:.2f}"
    )

    st.write(
        "Nhận xét: Đây là nhóm mẫu có các đặc điểm hóa học "
        "được mô tả bởi những giá trị trung bình phía trên."
    )


# =====================================================
# 10. DANH GIA K-MEANS
# =====================================================

st.markdown(
    '<div class="section">📐 7. ĐÁNH GIÁ K-MEANS</div>',
    unsafe_allow_html=True
)

if 1 < len(np.unique(clusters)) < len(X_scaled):

    sil = silhouette_score(X_scaled, clusters)

    db = davies_bouldin_score(X_scaled, clusters)

    ch = calinski_harabasz_score(X_scaled, clusters)

    metrics_df = pd.DataFrame({
        "Chỉ số": [
            "Inertia",
            "Silhouette Score",
            "Davies-Bouldin Index",
            "Calinski-Harabasz Score"
        ],
        "Giá trị": [
            kmeans.inertia_,
            sil,
            db,
            ch
        ],
        "Ý nghĩa": [
            "Độ chặt của cụm",
            "Độ tách biệt cụm",
            "Mức độ tương đồng giữa cụm",
            "Mức độ phân tách cụm"
        ]
    })

    st.dataframe(metrics_df, use_container_width=True)

else:
    st.warning(
        "Không thể tính đầy đủ chỉ số đánh giá "
        "với số lượng cụm hiện tại."
    )


# =====================================================
# 11. PCA TRUC QUAN HOA
# =====================================================

st.markdown(
    '<div class="section">🎨 8. TRỰC QUAN HÓA CỤM</div>',
    unsafe_allow_html=True
)

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(10, 6))

scatter = ax.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    cmap="viridis",
    s=90,
    alpha=0.8
)

centers_pca = pca.transform(kmeans.cluster_centers_)

ax.scatter(
    centers_pca[:, 0],
    centers_pca[:, 1],
    marker="X",
    s=250,
    c="red",
    label="Tâm cụm"
)

ax.set_title("PHÂN CỤM K-MEANS - PCA")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.legend()

fig.colorbar(scatter, ax=ax, label="Cluster")

st.pyplot(fig)
plt.close(fig)

st.caption(
    f"PCA giải thích tổng cộng "
    f"{sum(pca.explained_variance_ratio_) * 100:.2f}% "
    "phương sai của dữ liệu."
)


# =====================================================
# 12. HUAN LUYEN RANDOM FOREST
# =====================================================

st.markdown(
    '<div class="section">🌳 9. HUẤN LUYỆN RANDOM FOREST</div>',
    unsafe_allow_html=True
)

st.write("""
Random Forest sử dụng 4 thuộc tính hóa học để dự đoán
cột quality của bộ dữ liệu.
""")

class_counts = y.value_counts()

n_classes = y.nunique()

# Chỉ phân tầng khi đủ mẫu cho cả Train và Test
test_count = int(np.ceil(len(y) * 0.25))

can_stratify = (
    class_counts.min() >= 2
    and test_count >= n_classes
    and len(y) - test_count >= n_classes
)

stratify_value = y if can_stratify else None

if not can_stratify:
    st.warning(
        "Một số lớp có ít mẫu. Dữ liệu được chia "
        "Train/Test mà không phân tầng."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=stratify_value
)

# Pipeline tránh rò rỉ dữ liệu
rf_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("forest", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        max_features="sqrt"
    ))
])

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

st.success("Huấn luyện Random Forest thành công!")

st.write("### Kích thước dữ liệu")

a, b = st.columns(2)

a.metric("Dữ liệu Train", len(X_train))
b.metric("Dữ liệu Test", len(X_test))


# =====================================================
# 13. DANH GIA RANDOM FOREST
# =====================================================

st.markdown(
    '<div class="section">📊 10. ĐÁNH GIÁ RANDOM FOREST</div>',
    unsafe_allow_html=True
)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

a, b, c, d = st.columns(4)

a.metric("Accuracy", f"{accuracy:.4f}")
b.metric("Precision", f"{precision:.4f}")
c.metric("Recall", f"{recall:.4f}")
d.metric("F1-score", f"{f1:.4f}")

st.write("### Ma trận nhầm lẫn")

labels = np.sort(y.unique())

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

fig, ax = plt.subplots(figsize=(8, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    ax=ax
)

ax.set_xlabel("Dự đoán")
ax.set_ylabel("Thực tế")
ax.set_title("CONFUSION MATRIX - RANDOM FOREST")

st.pyplot(fig)
plt.close(fig)

st.write("### Báo cáo phân loại")

report = classification_report(
    y_test,
    y_pred,
    labels=labels,
    zero_division=0,
    output_dict=True
)

st.dataframe(
    pd.DataFrame(report).transpose()
)


# =====================================================
# 14. DU DOAN DU LIEU MOI
# =====================================================

st.markdown(
    '<div class="section">🔮 11. DỰ ĐOÁN DỮ LIỆU MỚI</div>',
    unsafe_allow_html=True
)

st.write(
    "Nhập 4 thuộc tính hóa học để dự đoán "
    "chất lượng rượu vang và xác định cụm."
)

medians = X.median()
mins = X.min()
maxs = X.max()

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        new_acidity = st.number_input(
            "Fixed acidity",
            min_value=float(mins["fixed acidity"]),
            max_value=float(maxs["fixed acidity"]),
            value=float(medians["fixed acidity"])
        )

        new_sulfur = st.number_input(
            "Total sulfur dioxide",
            min_value=float(mins["total sulfur dioxide"]),
            max_value=float(maxs["total sulfur dioxide"]),
            value=float(medians["total sulfur dioxide"])
        )

    with col2:

        new_ph = st.number_input(
            "pH",
            min_value=float(mins["pH"]),
            max_value=float(maxs["pH"]),
            value=float(medians["pH"])
        )

        new_alcohol = st.number_input(
            "Alcohol",
            min_value=float(mins["alcohol"]),
            max_value=float(maxs["alcohol"]),
            value=float(medians["alcohol"])
        )

    submitted = st.form_submit_button(
        "🔍 DỰ ĐOÁN",
        use_container_width=True
    )

if submitted:

    new_data = pd.DataFrame([{
        "fixed acidity": new_acidity,
        "total sulfur dioxide": new_sulfur,
        "pH": new_ph,
        "alcohol": new_alcohol
    }])

    # Dự đoán chất lượng bằng Random Forest
    predicted_quality = rf_model.predict(new_data)[0]

    # Xác định cụm bằng K-means
    new_imputed = imputer.transform(new_data)

    new_scaled = scaler.transform(new_imputed)

    predicted_cluster = kmeans.predict(new_scaled)[0]

    probabilities = rf_model.predict_proba(new_data)[0]

    confidence = probabilities.max() * 100

    st.success("Đã hoàn tất dự đoán!")

    a, b = st.columns(2)

    a.metric(
        "🔵 Cụm K-means",
        f"Cluster {predicted_cluster}"
    )

    b.metric(
        "🌳 Quality dự đoán",
        str(predicted_quality)
    )

    st.metric(
        "Xác suất cao nhất trong các lớp dự đoán",
        f"{confidence:.2f}%"
    )

    st.write("### Dữ liệu vừa nhập")
    st.dataframe(new_data)

    st.info(
        "Lưu ý: Xác suất Random Forest là mức độ tin cậy "
        "theo phân phối dự đoán của mô hình, không phải "
        "bảo đảm dự đoán đúng."
    )


# =====================================================
# 15. KET THUC
# =====================================================

st.divider()

st.markdown("""
<div style="text-align:center; color:#64748b; padding:20px;">
    🍷 HỆ THỐNG PHÂN CỤM VÀ DỰ ĐOÁN RƯỢU VANG<br>
    MACHINE LEARNING - K-MEANS & RANDOM FOREST
</div>
""", unsafe_allow_html=True)
