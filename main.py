import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# KONFIGURASI HALAMAN
st.set_page_config(page_title="Dashboard Portofolio & LQ45", layout="wide")

# LOAD DATASET
@st.cache_data
def load_data_keuangan():
    df = pd.read_csv("dataset_keuangan_final.csv")
    return df

@st.cache_data
def load_data_saham():
    # df = pd.read_csv("lq45_feature_engineering.csv")
    df = pd.read_parquet("lq45_feature_engineering.parquet")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df_uang = load_data_keuangan()
df_saham = load_data_saham()

# SIDEBAR NAVIGASI
st.sidebar.title("Dashboard Analysis")
pilihan_menu = st.sidebar.radio(
    "Pilih Halaman Analisis:",
    ("Beranda", "Profil Keuangan", "Saham LQ45", "Insight Terintegrasi")
)

st.sidebar.divider()
st.sidebar.info("Sistem ini terhubung langsung dengan dataset Keuangan (3.000 users) dan pergerakan Saham LQ45.")


# KONTEN HALAMAN BERDASARKAN PILIHAN
# HALAMAN BERANDA
if pilihan_menu == "Beranda":
    st.title("Sistem Analisis Profil Risiko & Peluang Investasi")
    st.markdown("""
    
    Dashboard ini mengintegrasikan dua analisis utama:
    1. **Data Finansial Internal:** Membedah kondisi kesehatan keuangan pengguna (dummy data)dan mengelompokkan profil risiko berdasarkan K-Means Clustering.
    2. **Data Pasar Saham (LQ45):** Menganalisis pergerakan saham LQ45 rentang tahun (2015-2025) di Indonesia sebagai rekomendasi investasi.
    
    *Silakan klik menu di sebelah kiri untuk mulai menjelajahi hasil analisiskami.*
    """)
    
    col1, col2 = st.columns(2)
    col1.metric("Total User Teranalisis (Keuangan)", f"{len(df_uang):,} User")
    col2.metric("Total Record Harga Saham (LQ45)", f"{len(df_saham):,} Baris Data")

# HALAMAN EDA KEUANGAN
elif pilihan_menu == "Profil Keuangan":

    st.title("Finansial & Clustering K-Means")
    
    col1, col2 = st.columns(2)
    
    # 1. DONUT CHART PERSENTASE CLUSTER (KIRI)
    st.subheader("1. Persentase Profil Investor")
    st.write("Proporsi pengguna berdasarkan hasil clustering algoritma K-Means.")
    
    # Menghitung porsi masing-masing kelompok
    df_pie = df_uang['status_kesiapan_invest'].value_counts().reset_index()
    df_pie.columns = ['Status Kesiapan', 'Jumlah']
    
    warna_cluster = {
        'Belum Siap Investasi': "#8dcf67",         
        'Siap Investasi (Konservatif)': '#f39c12', 
        'Siap Investasi (Agresif)': "#529fcc"      
    }
    
    fig_pie = px.pie(df_pie, values='Jumlah', names='Status Kesiapan', hole=0.45, 
                     color='Status Kesiapan', color_discrete_map=warna_cluster)
    
    # Menampilkan persentase dengan jelas di dalam chart
    fig_pie.update_traces(textposition='inside', textinfo='percent+label', 
                          showlegend=False) 
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("Dari total 3000 user (dataset), mayoritas berada di cluster 'Belum Siap Investasi' dengan proporsi 41,8%, diikuti oleh 'Siap Investasi (Konservatif)' 30,3%, dan sisanya 'Siap Investasi (Agresif)' 27,9%. Pembagiajn cluster tersebut adalah dari hasil algoritma K-Means yang mempertimbangkan fitur-fitur finansial utama seperti skor dana darurat, nominal investasi, dan rasio utang terhadap pemasukan.") 
    st.divider()

    # 2. KARAKTERISTIK FINANSIAL 
    st.subheader("2. Karakteristik Utama Tiap Profil")
    st.write("Perbandingan rata-rata fitur penentu dari setiap kelompok.")
    
    # Menghitung rata-rata untuk setiap fitur berdasarkan cluster
    df_karakter = df_uang.groupby('status_kesiapan_invest')[['score_emergency', 'investment_amount', 'debt_to_income_ratio']].mean().reset_index()
    
    # Grafik 1: Skor Dana Darurat
    fig_score = px.bar(df_karakter, x='status_kesiapan_invest', y='score_emergency', 
                       color='status_kesiapan_invest', color_discrete_map=warna_cluster)
    fig_score.update_layout(title="Rata-rata Skor Dana Darurat", showlegend=False, 
                            xaxis_title="Profil Investor", yaxis_title="Skor") 
    st.plotly_chart(fig_score, use_container_width=True)
    
    # Grafik 2: Nominal Investasi
    fig_invest = px.bar(df_karakter, x='status_kesiapan_invest', y='investment_amount', 
                        color='status_kesiapan_invest', color_discrete_map=warna_cluster)
    fig_invest.update_layout(title="Rata-rata Nominal Investasi", showlegend=False, 
                             xaxis_title="Profil Investor", yaxis_title="Nominal (Rp)")
    st.plotly_chart(fig_invest, use_container_width=True)
    
    # Grafik 3: Rasio Utang 
    fig_debt = px.bar(df_karakter, x='status_kesiapan_invest', y='debt_to_income_ratio', 
                      color='status_kesiapan_invest', color_discrete_map=warna_cluster)
    fig_debt.update_layout(title="Rata-rata Rasio Utang", showlegend=False, 
                           xaxis_title="Profil Investor", yaxis_title="Rasio DTI")
    st.plotly_chart(fig_debt, use_container_width=True)
    
    st.divider() 

    # 3. SCATTER PLOT RASIO UTANG
    st.subheader("3. Analisis Beban Utang vs Pemasukan")
    st.write("Sebaran pengguna berdasarkan rasio utang mereka. Algoritma menempatkan batas tegas bagi pengguna yang utangnya terlalu besar.")
    
    fig_scatter = px.scatter(df_uang, x='monthly_income', y='debt_to_income_ratio', 
                             color='status_kesiapan_invest', color_discrete_map=warna_cluster,
                             opacity=0.7, template='plotly_white')
    
    # Menambahkan garis batas horizontal (Batas Aman Utang 40%)
    fig_scatter.add_hline(y=0.40, line_dash="dash", line_color="red", annotation_text="Batas Aman Rasio Utang (40%)")
    fig_scatter.update_layout(xaxis_title="Pendapatan Bulanan (Rp)", yaxis_title="Debt to Income Ratio")
    st.plotly_chart(fig_scatter, use_container_width=True)

    
# HALAMAN EDA SAHAM 
elif pilihan_menu == "Saham LQ45":
    st.title("Pergerakan Saham LQ45")
    
    # Filter interaktif untuk memilih Ticker Saham
    daftar_saham = sorted(df_saham['Ticker'].unique())
    saham_terpilih = st.selectbox("Silakan Pilih Kode Saham di bawah ini untuk Dianalisis:", daftar_saham)
    
    df_saham_filter = df_saham[df_saham['Ticker'] == saham_terpilih]
    
    st.subheader(f"Tren Harga Penutupan (Close Price) - {saham_terpilih}")
    fig_line = px.line(df_saham_filter, x='Date', y='Close', template='plotly_white')
    # Menambahkan slider waktu di bawah grafik
    fig_line.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig_line, use_container_width=True)
    
    st.subheader(f"Volume Perdagangan Historis - {saham_terpilih}")
    fig_bar = go.Figure(data=[
        go.Bar(
            x=df_saham_filter['Date'], 
            y=df_saham_filter['Volume'], 
            marker_color='#529fcc' 
        )
    ])
    
    fig_bar.update_layout(
        xaxis_title="Tanggal",
        yaxis_title="Volume"
    )
                     
    # Tampilkan dengan theme=None
    st.plotly_chart(fig_bar, use_container_width=True)
    
    st.divider()
    
    # 1. Tampilan Risk vs Return (Untuk Semua Saham)
    st.subheader("1. Peta Risiko vs Keuntungan (Risk-Return Tradeoff)")
    st.write("Memetakan seluruh saham LQ45 berdasarkan fluktuasi harga bulanan dan rata-rata keuntungan harian.")
    
    # Mengambil data terakhir dari setiap Ticker untuk plot Scatter
    df_latest = df_saham.drop_duplicates(subset=['Ticker'], keep='last')
    
    fig_scatter = px.scatter(df_latest, x='Volatility', y='Log_Return', 
                             color='Risk_Category', hover_name='Ticker',
                             template='plotly_white', size_max=15,
                             color_discrete_map={'Low': '#2ecc71', 'Medium': '#f1c40f', 'High': '#e74c3c'})
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.divider()

    # 2. Analisis Teknikal Saham Spesifik (Moving Average)
    st.subheader("2. Analisis Teknikal: Moving Average & Harga")
    daftar_saham = sorted(df_saham['Ticker'].unique())
    saham_terpilih = st.selectbox("Silakan Pilih Kode Saham:", daftar_saham)
    
    df_saham_filter = df_saham[df_saham['Ticker'] == saham_terpilih]
    
    # Menggunakan Graph Objects plotly untuk menggabungkan banyak garis
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=df_saham_filter['Date'], y=df_saham_filter['Close'], mode='lines', name='Close Price', line=dict(color='green', width=2)))
    fig_line.add_trace(go.Scatter(x=df_saham_filter['Date'], y=df_saham_filter['MA20'], mode='lines', name='MA 20 (Jangka Pendek)', line=dict(color='blue', width=1.5)))
    fig_line.add_trace(go.Scatter(x=df_saham_filter['Date'], y=df_saham_filter['MA30'], mode='lines', name='MA 50 (Jangka Menengah)', line=dict(color='red', width=1.5)))
    
    fig_line.update_layout(template='plotly_white', xaxis_rangeslider_visible=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_line, use_container_width=True)

# HALAMAN INSIGHT TERINTEGRASI
elif pilihan_menu == "Insight Terintegrasi":
    st.title("Kesimpulan & Rekomendasi")
    
    # 1. PILIH PROFIL (Simulasi User Login)
    st.markdown("Simulasi Pengguna")
    profil_pilihan = st.selectbox("Pilih Cluster Profil Investor:", df_uang['status_kesiapan_invest'].unique())
    df_profil = df_uang[df_uang['status_kesiapan_invest'] == profil_pilihan]
    
    st.divider()
    
    # 2. DIAGNOSA KEUANGAN 
    st.markdown("Diagnosa Kesehatan Keuangan setiap Cluster")
    st.write("Berdasarkan data rata-rata profil 3000 user berikut adalah kondisi keuangan untuk setiap cluster:")
    
    # Menghitung rata-rata untuk ditampilkan di metrik
    rata_gaji = df_profil['monthly_income'].mean()
    rata_utang = df_profil['debt_to_income_ratio'].mean() * 100
    rata_invest = df_profil['investment_amount'].mean()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Pemasukan Bulanan", f"Rp {rata_gaji:,.0f}")
    col2.metric("Beban Utang / Cicilan", f"{rata_utang:.1f}% dari gaji")
    col3.metric("Kapasitas Investasi", f"Rp {rata_invest:,.0f}")

    st.markdown("<br>", unsafe_allow_html=True) # Spasi kosong

    # Mengambil data saham terbaru agar rekomendasinya dinamis (bukan hardcoded)
    df_saham_terbaru = df_saham.drop_duplicates(subset=['Ticker'], keep='last')
    
    # 3. REKOMENDASI DINAMIS & ACTION PLAN
    st.markdown("Rencana Aksi & Rekomendasi Sistem")
    
    if "Agresif" in profil_pilihan:
        st.success("**KESIMPULAN: KONDISI FINANSIAL SANGAT PRIMA!**")
        st.write("Cluster inimemiliki dana darurat yang melimpah dan beban utang yang sangat aman. Cluster ini sangat siap untuk menumbuhkan kekayaan di instrumen saham dengan tingkat pertumbuhan (dan risiko) yang lebih tinggi.")
        
        st.markdown("Rekomendasi Saham (Risk Class: Medium & High)")
        # Filter dinamis dari data LQ45
        saham_agresif = df_saham_terbaru[df_saham_terbaru['Risk_Category'].isin(['Medium', 'High'])]
        saham_agresif = saham_agresif[['Ticker', 'Close', 'Volatility', 'Risk_Category']].sort_values(by='Volatility', ascending=False).head(5)
        
        # Merapikan nama kolom untuk orang awam
        saham_agresif.columns = ['Kode Saham', 'Harga Terakhir (Rp)', 'Tingkat Fluktuasi (Volatilitas)', 'Kelas Risiko']
        st.dataframe(saham_agresif, use_container_width=True, hide_index=True)
        st.caption("*Tips: Saham di atas berpotensi memberikan keuntungan besar, namun harganya bisa naik-turun dengan cepat.*")

    elif "Konservatif" in profil_pilihan:
        st.info("**KESIMPULAN: KONDISI FINANSIAL STABIL & AMAN**")
        st.write("Cluster ini sudah siap berinvestasi, namun kami menyarankan untuk bermain aman. Fokuslah pada perusahaan besar yang rajin membagikan dividen dan harganya tidak terlalu bergejolak.")
        
        st.markdown("Rekomendasi Saham (Risk Class: Low)")
        # Filter dinamis dari data LQ45
        saham_konservatif = df_saham_terbaru[df_saham_terbaru['Risk_Category'] == 'Low']
        saham_konservatif = saham_konservatif[['Ticker', 'Close', 'Volatility', 'Risk_Category']].sort_values(by='Volatility', ascending=True).head(5)
        
        saham_konservatif.columns = ['Kode Saham', 'Harga Terakhir (Rp)', 'Tingkat Fluktuasi (Volatilitas)', 'Kelas Risiko']
        st.dataframe(saham_konservatif, use_container_width=True, hide_index=True)
        st.caption("*Tips: Saham di atas pergerakannya lebih lambat, sangat cocok untuk tabungan jangka panjang yang tenang.*")

    else:
        st.error("**KESIMPULAN: FOKUS PERBAIKI FONDASI KEUANGAN DULU!**")
        st.write("Sistem mendeteksi bahwa beban utang Cluster ini masih terlalu tinggi atau dana daruratnya belum cukup. Memaksa beli saham saat ini sangat berbahaya bagi psikologis keuangan Cluster ini.")
        
        st.markdown("Langkah yang Harus Anda Lakukan:")
        st.markdown("""
        1. **Stop Beli Saham**
        2. **Lunasi Utang:** Fokuskan uang ekstra untuk melunasi cicilan hingga di bawah 40% dari gaji.
        3. **Bangun Dana Darurat:** Jika ingin menabung, gunakan **Reksa Dana Pasar Uang (RDPU)** yang risikonya mendekati nol (0).
        """)
        
        # Tampilkan visual peringatan
        st.warning("Ingat: Investasi terbaik saat ini adalah melunasi utang berbunga tinggi!")
