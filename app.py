import re
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer

# ----------------------------------------------------
# Konfigurasi dasar app
# ----------------------------------------------------
st.set_page_config(
    page_title="Rekomendasi Judul Hibah Penelitian",
    layout="wide",
)

st.title("📑 Rekomendasi Judul Hibah Penelitian dari Publikasi Scopus")
st.markdown(
    "Aplikasi ini membaca data publikasi dari file Excel, "
    "lalu menghasilkan beberapa usulan **judul hibah + abstrak singkat** "
    "per dosen berdasarkan rekam jejak publikasinya."
)

# ----------------------------------------------------
# Helper functions
# ----------------------------------------------------
def guess_column(columns: List[str], keywords: List[str]) -> Optional[str]:
    """
    Mencari nama kolom yang paling mirip dengan keyword-keyword tertentu.
    Contoh: keywords = ['kode', 'lecturer', 'dosen']
    """
    cols_lower = {c.lower(): c for c in columns}
    for key in keywords:
        for c_low, c_orig in cols_lower.items():
            if key in c_low:
                return c_orig
    return None


def generate_topics_from_publications(texts: List[str], top_k: int = 5) -> List[str]:
    """
    Mengambil topik-topik utama dari list judul/abstrak menggunakan TF-IDF n-gram.
    """
    cleaned_texts = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
    if not cleaned_texts:
        return []

    vectorizer = TfidfVectorizer(
        ngram_range=(2, 3),
        max_features=80,
        stop_words="english",  # mayoritas judul Scopus sering English
    )
    X = vectorizer.fit_transform(cleaned_texts)
    scores = np.asarray(X.sum(axis=0)).ravel()
    feature_names = np.array(vectorizer.get_feature_names_out())

    # Urutkan fitur berdasarkan skor TF-IDF total
    sorted_idx = scores.argsort()[::-1]

    topics = []
    for idx in sorted_idx:
        phrase = feature_names[idx]
        phrase_clean = re.sub(r"\b\d+\b", " ", phrase)
        phrase_clean = re.sub(r"\s+", " ", phrase_clean).strip()
        if len(phrase_clean) < 5:
            continue
        # Hindari duplikat yang terlalu mirip
        if phrase_clean.lower() not in [t.lower() for t in topics]:
            topics.append(phrase_clean)
        if len(topics) >= top_k:
            break

    return topics


def make_title_from_topic(topic: str, variant: int = 0) -> str:
    """
    Membuat judul hibah berbasis topik tertentu.
    Bahasa Indonesia, tapi fleksibel untuk topik English.
    """
    t = topic.strip()
    # Biar agak rapi untuk judul
    t_title = re.sub(r"\s+", " ", t).title()

    templates = [
        f"Pengembangan Lanjutan Riset Mengenai {t_title}",
        f"Model dan Implementasi {t_title} Untuk Peningkatan Kinerja dan Daya Saing",
        f"Optimalisasi {t_title} Melalui Pendekatan Terintegrasi",
        f"Strategi Kolaboratif Dalam Penerapan {t_title} di Konteks Indonesia",
        f"Hilirisasi dan Penerapan {t_title} Untuk Menjawab Tantangan Industri dan Masyarakat",
    ]

    return templates[variant % len(templates)]


def make_abstract_from_topic(topic: str, focus: str = "") -> str:
    """
    Membuat abstrak singkat (±3–4 kalimat) berbasis topik.
    """
    t = topic.strip()
    if focus:
        fokus_text = f" dengan fokus pada {focus}"
    else:
        fokus_text = ""

    abstract = (
        f"Penelitian ini bertujuan untuk mengembangkan riset lanjutan pada topik {t}{fokus_text}, "
        f"yang sebelumnya telah menjadi salah satu fokus publikasi peneliti. "
        f"Studi ini akan memetakan tren penelitian terkini terkait {t}, "
        f"merumuskan model atau kerangka kerja yang lebih komprehensif, "
        f"serta menguji penerapannya pada konteks studi kasus yang relevan. "
        f"Metode yang digunakan dapat mencakup analisis kuantitatif maupun kualitatif, "
        f"disesuaikan dengan karakteristik data dan tujuan penelitian. "
        f"Diharapkan, hasil penelitian ini dapat memberikan kontribusi teoritis maupun praktis, "
        f"serta membuka peluang kolaborasi lebih luas dan hilirisasi hasil riset."
    )
    return abstract


def build_grant_ideas(
    topics: List[str],
    n_ideas: int,
    focus: str = "",
) -> List[Dict[str, str]]:
    """
    Dari list 'topics', bangun n_ideas ide hibah:
    masing-masing berisi judul, abstrak, dan topik kunci.
    """
    ideas = []
    if not topics:
        topics = ["pengembangan riset dan publikasi"]

    for i in range(n_ideas):
        topic = topics[i % len(topics)]
        title = make_title_from_topic(topic, variant=i)
        abstract = make_abstract_from_topic(topic, focus=focus)
        ideas.append(
            {
                "judul_hibah": title,
                "abstrak_hibah": abstract,
                "topik_kunci": topic,
            }
        )
    return ideas


# ----------------------------------------------------
# 1. Upload & mapping kolom
# ----------------------------------------------------
st.sidebar.header("1️⃣ Upload Data & Mapping Kolom")

uploaded_file = st.sidebar.file_uploader(
    "Upload file Excel publikasi Scopus",
    type=["xlsx", "xls"],
    help="Gunakan file yang berisi data publikasi multi-dosen (mis. semua tahun 2014–2022).",
)

if uploaded_file is None:
    st.info("⬅️ Silakan upload file Excel publikasi terlebih dahulu.")
    st.stop()

# Baca Excel
try:
    df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Gagal membaca file Excel: {e}")
    st.stop()

if df.empty:
    st.warning("File yang diupload kosong.")
    st.stop()

st.expander("Lihat sampel data").dataframe(df.head())

columns = df.columns.tolist()

# Tebak kolom secara otomatis
default_kode = guess_column(columns, ["kode dosen", "kd dosen", "kode_dosen", "lecturer id"])
default_nama = guess_column(columns, ["nama dosen", "author", "nama"])
default_judul = guess_column(columns, ["judul", "title", "article title"])
default_abstrak = guess_column(columns, ["abstrak", "abstract"])
default_tahun = guess_column(columns, ["tahun", "year"])

kode_col = st.sidebar.selectbox(
    "Kolom Kode Dosen",
    options=columns,
    index=columns.index(default_kode) if default_kode in columns else 0,
)

nama_col = st.sidebar.selectbox(
    "Kolom Nama Dosen (opsional)",
    options=["(Tidak ada)"] + columns,
    index=(columns.index(default_nama) + 1) if default_nama in columns else 0,
)

judul_col = st.sidebar.selectbox(
    "Kolom Judul Artikel",
    options=columns,
    index=columns.index(default_judul) if default_judul in columns else 0,
)

abstrak_col = st.sidebar.selectbox(
    "Kolom Abstrak Artikel (opsional)",
    options=["(Tidak ada)"] + columns,
    index=(columns.index(default_abstrak) + 1) if default_abstrak in columns else 0,
)

tahun_col = st.sidebar.selectbox(
    "Kolom Tahun Publikasi (opsional)",
    options=["(Tidak ada)"] + columns,
    index=(columns.index(default_tahun) + 1) if default_tahun in columns else 0,
)

abstrak_col_real = None if abstrak_col == "(Tidak ada)" else abstrak_col
tahun_col_real = None if tahun_col == "(Tidak ada)" else tahun_col
nama_col_real = None if nama_col == "(Tidak ada)" else nama_col

# ----------------------------------------------------
# 2. Pilih dosen berdasarkan Kode Dosen
# ----------------------------------------------------
st.sidebar.header("2️⃣ Pilih Dosen")

kode_list = sorted(set(df[kode_col].dropna().astype(str)))
if not kode_list:
    st.error(f"Kolom '{kode_col}' tidak memiliki nilai Kode Dosen.")
    st.stop()

selected_kode = st.sidebar.selectbox(
    "Pilih Kode Dosen",
    options=kode_list,
)

dosen_df = df[df[kode_col].astype(str) == str(selected_kode)].copy()

if dosen_df.empty:
    st.warning("Tidak ada publikasi untuk Kode Dosen tersebut.")
    st.stop()

# Ambil nama dosen (kalau ada)
nama_dosen = None
if nama_col_real is not None and nama_col_real in dosen_df.columns:
    # Ambil nama yang paling sering muncul
    nama_dosen = (
        dosen_df[nama_col_real]
        .dropna()
        .astype(str)
        .value_counts()
        .index[0]
        if not dosen_df[nama_col_real].dropna().empty
        else None
    )

# Info ringkas dosen
col_info, col_pub = st.columns([1, 2])

with col_info:
    st.subheader("👤 Profil Singkat Dosen")
    st.markdown(f"**Kode Dosen:** `{selected_kode}`")
    if nama_dosen:
        st.markdown(f"**Nama Dosen:** {nama_dosen}")
    st.markdown(f"**Jumlah Publikasi (baris data):** {len(dosen_df)}")

    if tahun_col_real and tahun_col_real in dosen_df.columns:
        try:
            years = pd.to_numeric(dosen_df[tahun_col_real], errors="coerce").dropna()
            if not years.empty:
                st.markdown(
                    f"**Rentang Tahun Publikasi:** "
                    f"{int(years.min())} – {int(years.max())}"
                )
        except Exception:
            pass

with col_pub:
    st.subheader("📚 Daftar Publikasi Dosen (ringkas)")
    st.dataframe(
        dosen_df[[c for c in [judul_col, tahun_col_real] if c is not None]],
        use_container_width=True,
        height=250,
    )

# ----------------------------------------------------
# 3. Generate rekomendasi judul hibah + abstrak
# ----------------------------------------------------
st.subheader("🧠 Generate Rekomendasi Judul Hibah")

col_left, col_right = st.columns([1.2, 1])

with col_left:
    n_ideas = st.slider(
        "Jumlah usulan judul hibah yang ingin dihasilkan",
        min_value=1,
        max_value=10,
        value=3,
    )
    top_k_topics = st.slider(
        "Jumlah topik utama yang dianalisis dari publikasi",
        min_value=3,
        max_value=15,
        value=7,
        help="Digunakan untuk menangkap kata/frasa penting dari judul/abstrak.",
    )
    focus_text = st.text_input(
        "Fokus hibah (opsional)",
        placeholder="Misal: sustainability, pendidikan tinggi, UMKM, kesehatan, dsb.",
        help="Ini akan disisipkan di abstrak jika diisi.",
    )

with col_right:
    st.markdown("**Sumber teks analisis:**")
    use_abstract = st.checkbox(
        "Sertakan kolom abstrak (jika tersedia)",
        value=True if abstrak_col_real else False,
        disabled=(abstrak_col_real is None),
    )
    st.caption(
        "Sistem akan menggabungkan judul (dan abstrak jika dipilih) untuk "
        "mencari topik-topik utama yang sering muncul."
    )

generate_button = st.button("🔍 Generate rekomendasi judul hibah", type="primary")

if generate_button:
    # Kumpulkan teks semua publikasi dosen
    texts = []
    for _, row in dosen_df.iterrows():
        title_text = str(row.get(judul_col, "")) if pd.notna(row.get(judul_col, "")) else ""
        abstract_text = ""
        if use_abstract and abstrak_col_real:
            abstract_text = (
                str(row.get(abstrak_col_real, "")) if pd.notna(row.get(abstrak_col_real, "")) else ""
            )
        combined = (title_text + " " + abstract_text).strip()
        if combined:
            texts.append(combined)

    if not texts:
        st.error(
            "Tidak ada teks judul/abstrak yang bisa dianalisis. "
            "Pastikan kolom judul (dan abstrak, jika dipakai) berisi data."
        )
    else:
        with st.spinner("Menganalisis topik dari publikasi dosen..."):
            topics = generate_topics_from_publications(texts, top_k=top_k_topics)
            ideas = build_grant_ideas(topics, n_ideas=n_ideas, focus=focus_text)

        st.session_state["grant_ideas"] = ideas
        st.session_state["grant_topics"] = topics

# Tampilkan hasil jika sudah ada di session_state
ideas = st.session_state.get("grant_ideas", [])

if ideas:
    st.markdown("---")
    st.subheader("📌 Rekomendasi Judul Hibah & Abstrak")

    topics = st.session_state.get("grant_topics", [])
    if topics:
        st.markdown("**Topik-topik utama yang terdeteksi dari publikasi dosen:**")
        st.write(", ".join(topics))

    for idx, idea in enumerate(ideas, start=1):
        with st.expander(f"💡 Ide #{idx}: {idea['judul_hibah']}", expanded=(idx == 1)):
            st.markdown(f"**Judul Hibah:** {idea['judul_hibah']}")
            st.markdown(f"**Topik Kunci:** {idea['topik_kunci']}")
            st.markdown("**Abstrak Singkat (Draft):**")
            st.write(idea["abstrak_hibah"])

    # Optional: download sebagai Excel/CSV
    df_ideas = pd.DataFrame(ideas)
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv = df_ideas.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download ide hibah (CSV)",
            data=csv,
            file_name=f"ide_hibah_{selected_kode}.csv",
            mime="text/csv",
        )
    with col_dl2:
        excel_buf = pd.ExcelWriter("temp.xlsx", engine="xlsxwriter")
        df_ideas.to_excel(excel_buf, index=False, sheet_name="GrantIdeas")
        excel_buf.close()
        with open("temp.xlsx", "rb") as f:
            st.download_button(
                "⬇️ Download ide hibah (Excel)",
                data=f,
                file_name=f"ide_hibah_{selected_kode}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

# ----------------------------------------------------
# 4. Template email ke dosen
# ----------------------------------------------------
st.markdown("---")
st.subheader("📧 Draft Email ke Dosen")

if "email_template" not in st.session_state:
    st.session_state["email_template"] = (
        "Yth. {NAMA_DOSEN},\n\n"
        "Berdasarkan rekam jejak publikasi Bapak/Ibu dengan Kode Dosen {KODE_DOSEN}, "
        "kami melihat adanya peluang untuk mengusulkan hibah penelitian lanjutan.\n\n"
        "Berikut salah satu usulan judul hibah yang disusun berdasarkan publikasi Bapak/Ibu:\n\n"
        "Judul Hibah:\n"
        "\"{JUDUL_HIBAH}\"\n\n"
        "Abstrak Singkat:\n"
        "{ABSTRAK_HIBAH}\n\n"
        "Jika Bapak/Ibu berkenan, kami dapat mendiskusikan lebih lanjut terkait skema hibah yang sesuai "
        "serta penyusunan proposal lengkapnya.\n\n"
        "Hormat kami,\n"
        "[Nama Anda]\n"
        "[Unit/Institusi]\n"
    )

st.markdown("**Placeholder yang tersedia di template email:**")
st.code(
    "{NAMA_DOSEN}, {KODE_DOSEN}, {JUDUL_HIBAH}, {ABSTRAK_HIBAH}, "
    "{TAHUN_AWAL}, {TAHUN_AKHIR}, {JUMLAH_PUBLIKASI}, {TOPIK_KUNCI}",
    language="text",
)

email_template = st.text_area(
    "Template email (bisa diedit, gunakan placeholder di atas):",
    value=st.session_state["email_template"],
    height=260,
)
st.session_state["email_template"] = email_template

if not ideas:
    st.info("Generate dulu ide hibah di atas supaya bisa dibuatkan draft email.")
else:
    # Pilih ide hibah mana yang mau dimasukkan ke email
    idea_options = [f"Ide #{i+1}: {idea['judul_hibah']}" for i, idea in enumerate(ideas)]
    selected_idea_idx = st.selectbox(
        "Pilih ide hibah untuk dimasukkan ke email:",
        options=list(range(len(idea_options))),
        format_func=lambda i: idea_options[i],
    )

    # Siapkan konteks untuk .format()
    selected_idea = ideas[selected_idea_idx]

    # Ambil info tahun dan jumlah publikasi
    tahun_awal, tahun_akhir = "", ""
    if tahun_col_real and tahun_col_real in dosen_df.columns:
        years = pd.to_numeric(dosen_df[tahun_col_real], errors="coerce").dropna()
        if not years.empty:
            tahun_awal, tahun_akhir = int(years.min()), int(years.max())

    context = {
        "NAMA_DOSEN": nama_dosen or "",
        "KODE_DOSEN": selected_kode,
        "JUDUL_HIBAH": selected_idea["judul_hibah"],
        "ABSTRAK_HIBAH": selected_idea["abstrak_hibah"],
        "TAHUN_AWAL": tahun_awal,
        "TAHUN_AKHIR": tahun_akhir,
        "JUMLAH_PUBLIKASI": len(dosen_df),
        "TOPIK_KUNCI": selected_idea["topik_kunci"],
    }

    if st.button("✉️ Generate draft email dengan placeholder terisi"):
        try:
            filled_email = email_template.format(**context)
            st.subheader("📨 Draft Email")
            st.text_area(
                "Silakan copy & edit seperlunya sebelum dikirim:",
                value=filled_email,
                height=320,
            )
        except KeyError as e:
            st.error(
                f"Template mengandung placeholder yang tidak dikenal: {e}. "
                f"Pastikan hanya menggunakan placeholder yang tercantum di atas."
            )
