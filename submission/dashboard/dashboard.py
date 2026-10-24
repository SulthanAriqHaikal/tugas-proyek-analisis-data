import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ==============================================================================
# KONFIGURASI HALAMAN & GAYA VISUAL (CLEAN ANALYTICAL UI)
# ==============================================================================
st.set_page_config(
    page_title="Sistem Analitik Kualitas Udara Beijing (2013–2017)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stMetric"] {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 6px;
            padding: 12px 16px;
        }
        div[data-testid="stMetricLabel"] > div {
            font-size: 0.82rem;
            font-weight: 600;
            color: #475569;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }
        div[data-testid="stMetricValue"] > div {
            font-size: 1.45rem;
            font-weight: 700;
            color: #0F172A;
        }
        h1, h2, h3 {
            color: #0F172A;
            font-family: 'Inter', -apple-system, sans-serif;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 1px solid #E2E8F0;
        }
        .stTabs [data-baseweb="tab"] {
            height: 42px;
            padding: 0 18px;
            font-weight: 600;
            color: #475569;
        }
        .stTabs [aria-selected="true"] {
            color: #0F172A;
            border-bottom: 2px solid #0F172A;
        }
    </style>
    """,
    unsafe_allow_html=True
)

sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['axes.linewidth'] = 0.8

# ==============================================================================
# KONSTANTA DOMAIN & METADATA STASIUN
# ==============================================================================
STATION_METADATA = {
    'Aotizhongxin': (39.982, 116.397, 'Urban'),
    'Dongsi': (39.929, 116.417, 'Urban'),
    'Guanyuan': (39.929, 116.339, 'Urban'),
    'Gucheng': (39.914, 116.184, 'Urban'),
    'Nongzhanguan': (39.937, 116.461, 'Urban'),
    'Tiantan': (39.886, 116.407, 'Urban'),
    'Wanliu': (39.987, 116.287, 'Urban'),
    'Wanshouxigong': (39.878, 116.352, 'Urban'),
    'Changping': (40.217, 116.230, 'Suburban'),
    'Dingling': (40.292, 116.220, 'Suburban'),
    'Huairou': (40.328, 116.628, 'Suburban'),
    'Shunyi': (40.127, 116.655, 'Suburban')
}

POLLUTANT_SPECS = {
    'PM2.5': {'label': 'PM2.5 (Partikulat Halus)', 'unit': 'µg/m³', 'threshold': 35.0, 'std_name': 'Baku Mutu Tahunan GB 3095-2012 (35 µg/m³)'},
    'PM10': {'label': 'PM10 (Partikulat Kasar)', 'unit': 'µg/m³', 'threshold': 70.0, 'std_name': 'Baku Mutu Tahunan GB 3095-2012 (70 µg/m³)'},
    'NO2': {'label': 'NO2 (Nitrogen Dioksida)', 'unit': 'µg/m³', 'threshold': 40.0, 'std_name': 'Baku Mutu Tahunan GB 3095-2012 (40 µg/m³)'},
    'SO2': {'label': 'SO2 (Sulfur Dioksida)', 'unit': 'µg/m³', 'threshold': 60.0, 'std_name': 'Baku Mutu Tahunan GB 3095-2012 (60 µg/m³)'},
    'CO': {'label': 'CO (Karbon Monoksida)', 'unit': 'µg/m³', 'threshold': 4000.0, 'std_name': 'Baku Mutu 24-Jam GB 3095-2012 (4000 µg/m³)'},
    'O3': {'label': 'O3 (Ozon Permukaan)', 'unit': 'µg/m³', 'threshold': 100.0, 'std_name': 'Baku Mutu 8-Jam Kelas I (100 µg/m³)'}
}

AQI_BINS = [0, 35, 75, 115, 150, 250, 2000]
AQI_LABELS = [
    'Good (0-35)',
    'Moderate (35-75)',
    'Unhealthy for Sensitive (75-115)',
    'Unhealthy (115-150)',
    'Very Unhealthy (150-250)',
    'Hazardous (>250)'
]
AQI_COLORS = ['#0F766E', '#EAB308', '#F97316', '#DC2626', '#7E22CE', '#450A0A']
COMPASS_ORDER = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des']


# ==============================================================================
# PIPELINE PEMUATAN & PEMROSESAN DATA OTOMATIS (SUPPORT SUB-FOLDER CLOUD & LOKAL)
# ==============================================================================
@st.cache_data(show_spinner="Memuat dan memvalidasi struktur dataset kualitas udara...")
def load_and_prepare_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)

    main_csv_paths = [
        os.path.join(base_dir, "main_data.csv"),
        "submission/dashboard/main_data.csv",
        "dashboard/main_data.csv",
        "main_data.csv"
    ]
    hourly_csv_paths = [
        os.path.join(base_dir, "hourly_profile.csv"),
        "submission/dashboard/hourly_profile.csv",
        "dashboard/hourly_profile.csv",
        "hourly_profile.csv"
    ]

    existing_main = next((p for p in main_csv_paths if os.path.exists(p)), None)
    existing_hourly = next((p for p in hourly_csv_paths if os.path.exists(p)), None)

    raw_patterns = [
        os.path.join(parent_dir, "data", "PRSA_Data_*.csv"),
        "submission/data/PRSA_Data_*.csv",
        "data/PRSA_Data_*.csv",
        "../data/PRSA_Data_*.csv",
        "PRSA_Data_*.csv"
    ]
    raw_files = []
    for pattern in raw_patterns:
        found = sorted(glob.glob(pattern))
        if found:
            raw_files = found
            break

    need_rebuild = True
    if existing_main is not None:
        sample_cols = pd.read_csv(existing_main, nrows=2).columns.tolist()
        if 'wd' in sample_cols and 'PM2.5_max' in sample_cols and existing_hourly is not None:
            need_rebuild = False

    if need_rebuild and len(raw_files) > 0:
        df_raw = pd.concat([pd.read_csv(f) for f in raw_files], ignore_index=True)
        df_raw['datetime'] = pd.to_datetime(df_raw[['year', 'month', 'day', 'hour']])
        df_raw = df_raw.sort_values(['station', 'datetime']).reset_index(drop=True)

        num_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
        for col in num_cols:
            df_raw[col] = df_raw.groupby('station')[col].transform(
                lambda x: x.interpolate(method='linear').ffill().bfill()
            )
        df_raw['wd'] = df_raw.groupby('station')['wd'].transform(lambda x: x.ffill().bfill())

        coord_df = pd.DataFrame.from_dict(
            STATION_METADATA, orient='index', columns=['lat', 'lon', 'zone']
        ).reset_index().rename(columns={'index': 'station'})
        df_raw = df_raw.merge(coord_df, on='station', how='left')

        daily_df = df_raw.groupby(
            [df_raw['datetime'].dt.date.rename('date'), 'year', 'month', 'station', 'zone', 'lat', 'lon']
        ).agg(
            **{
                'PM2.5': ('PM2.5', 'mean'),
                'PM2.5_max': ('PM2.5', 'max'),
                'PM2.5_p25': ('PM2.5', lambda x: np.percentile(x, 25)),
                'PM2.5_p75': ('PM2.5', lambda x: np.percentile(x, 75)),
                'PM10': ('PM10', 'mean'),
                'SO2': ('SO2', 'mean'),
                'NO2': ('NO2', 'mean'),
                'CO': ('CO', 'mean'),
                'O3': ('O3', 'mean'),
                'O3_max': ('O3', 'max'),
                'TEMP': ('TEMP', 'mean'),
                'PRES': ('PRES', 'mean'),
                'DEWP': ('DEWP', 'mean'),
                'RAIN': ('RAIN', 'sum'),
                'WSPM': ('WSPM', 'mean'),
                'wd': ('wd', lambda x: x.mode().iloc[0] if not x.mode().empty else 'NE')
            }
        ).reset_index()

        daily_df['AQI_Category'] = pd.cut(daily_df['PM2.5'], bins=AQI_BINS, labels=AQI_LABELS, right=True)

        hourly_df = df_raw.groupby(['station', 'zone', 'hour'])[
            ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'WSPM']
        ].mean().reset_index()

        try:
            daily_df.to_csv(os.path.join(base_dir, "main_data.csv"), index=False)
            hourly_df.to_csv(os.path.join(base_dir, "hourly_profile.csv"), index=False)
        except OSError:
            pass

        daily_df['date'] = pd.to_datetime(daily_df['date'])
        return daily_df, hourly_df

    elif existing_main is not None:
        daily_df = pd.read_csv(existing_main)
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        if 'AQI_Category' not in daily_df.columns:
            daily_df['AQI_Category'] = pd.cut(daily_df['PM2.5'], bins=AQI_BINS, labels=AQI_LABELS, right=True)
        if 'zone' not in daily_df.columns:
            coord_df = pd.DataFrame.from_dict(
                STATION_METADATA, orient='index', columns=['lat', 'lon', 'zone']
            ).reset_index().rename(columns={'index': 'station'})
            daily_df = daily_df.merge(coord_df, on='station', how='left')

        hourly_df = pd.read_csv(existing_hourly) if existing_hourly is not None else None
        return daily_df, hourly_df

    else:
        st.error(
            f"Berkas data tidak ditemukan di {base_dir}. Pastikan 'main_data.csv' atau folder 'data/' tersedia."
        )
        st.stop()


df, hourly_df = load_and_prepare_data()

# ==============================================================================
# PANEL KONTROL SAMPING (SIDEBAR PARAMETER & FILTER)
# ==============================================================================
with st.sidebar:
    st.markdown("### Parameter Analitik")
    selected_pollutant = st.selectbox(
        "Polutan Utama yang Dianalisis",
        options=list(POLLUTANT_SPECS.keys()),
        format_func=lambda x: POLLUTANT_SPECS[x]['label'],
        index=0
    )

    ma_window = st.select_slider(
        "Jendela Dekomposisi Moving Average (Hari)",
        options=[7, 14, 30, 60, 90],
        value=30
    )

    st.markdown("---")
    st.markdown("### Filter Spasial & Temporal")

    min_date = df['date'].min().date()
    max_date = df['date'].max().date()

    date_range = st.date_input(
        "Rentang Periode Observasi",
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

    zone_filter = st.radio(
        "Klasifikasi Zona Wilayah",
        options=["Seluruh Zona", "Urban", "Suburban"],
        horizontal=True
    )

    if zone_filter == "Seluruh Zona":
        station_pool = sorted(df['station'].unique().tolist())
    else:
        station_pool = sorted(df[df['zone'] == zone_filter]['station'].unique().tolist())

    selected_stations = st.multiselect(
        "Stasiun Pemantauan Aktif",
        options=station_pool,
        default=station_pool
    )

    st.markdown("---")
    st.caption(
        "Sumber Data: Beijing Multi-Site Air-Quality Dataset (Maret 2013 – Februari 2017). "
        "Standar acuan mengacu pada Baku Mutu Nasional GB 3095-2012."
    )

if len(date_range) == 2:
    start_dt, end_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
else:
    start_dt, end_dt = pd.to_datetime(min_date), pd.to_datetime(max_date)

if not selected_stations:
    st.warning("Pilih minimal satu stasiun pemantauan pada panel kontrol di sebelah kiri.")
    st.stop()

filtered_df = df[
    (df['date'] >= start_dt) &
    (df['date'] <= end_dt) &
    (df['station'].isin(selected_stations))
].copy()

if filtered_df.empty:
    st.warning("Tidak ada data untuk rentang tanggal dan stasiun yang dipilih.")
    st.stop()

spec = POLLUTANT_SPECS[selected_pollutant]
threshold_val = spec['threshold']
unit_str = spec['unit']

# ==============================================================================
# HEADER EKSEKUTIF & BARIS INDIKATOR KINERJA UTAMA (6 KPI CARDS)
# ==============================================================================
st.title("Sistem Analitik Kualitas Udara & Meteorologi Beijing")
st.markdown(
    f"**Parameter Aktif:** `{spec['label']}` | **Periode:** `{start_dt.strftime('%d %b %Y')}` – `{end_dt.strftime('%d %b %Y')}` | "
    f"**Cakupan:** `{len(selected_stations)} Stasiun ({len(filtered_df):,} hari-stasiun observasi)`"
)

mean_val = filtered_df[selected_pollutant].mean()
median_val = filtered_df[selected_pollutant].median()
iqr_val = filtered_df[selected_pollutant].quantile(0.75) - filtered_df[selected_pollutant].quantile(0.25)
max_row = filtered_df.loc[filtered_df[selected_pollutant].idxmax()]
exceed_pct = (filtered_df[selected_pollutant] > threshold_val).mean() * 100

urban_mean = df[(df['date'] >= start_dt) & (df['date'] <= end_dt) & (df['zone'] == 'Urban')][selected_pollutant].mean()
suburban_mean = df[(df['date'] >= start_dt) & (df['date'] <= end_dt) & (df['zone'] == 'Suburban')][selected_pollutant].mean()
urban_diff_pct = ((urban_mean - suburban_mean) / suburban_mean * 100) if suburban_mean > 0 else 0.0

kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

with kpi1:
    diff_thresh = ((mean_val - threshold_val) / threshold_val) * 100
    st.metric(
        f"Rata-Rata {selected_pollutant}",
        f"{mean_val:.2f} {unit_str}",
        delta=f"{diff_thresh:+.1f}% vs Baku Mutu ({threshold_val:.0f})",
        delta_color="inverse"
    )
with kpi2:
    st.metric(
        f"Median & IQR {selected_pollutant}",
        f"{median_val:.2f} {unit_str}",
        delta=f"IQR: {iqr_val:.1f} {unit_str}",
        delta_color="off"
    )
with kpi3:
    st.metric(
        "Puncak Harian Maksimum",
        f"{max_row[selected_pollutant]:.1f} {unit_str}",
        delta=f"{max_row['station']} ({max_row['date'].strftime('%b %Y')})",
        delta_color="off"
    )
with kpi4:
    st.metric(
        "Rasio Melampaui Baku Mutu",
        f"{exceed_pct:.1f}% Hari",
        delta=f"Ambang > {threshold_val:.0f} {unit_str}",
        delta_color="inverse" if exceed_pct > 50 else "off"
    )
with kpi5:
    st.metric(
        "Disparitas Urban vs Suburban",
        f"{urban_diff_pct:+.1f}%",
        delta=f"Urban {urban_mean:.1f} | Sub {suburban_mean:.1f}",
        delta_color="inverse"
    )
with kpi6:
    st.metric(
        "Rata-Rata Angin & Suhu",
        f"{filtered_df['WSPM'].mean():.2f} m/s",
        delta=f"Suhu Rata-Rata: {filtered_df['TEMP'].mean():.1f} °C",
        delta_color="off"
    )

st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# ARSITEKTUR 5 TAB ANALITIK
# ==============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Komparasi Spasial & Anomali Stasiun",
    "2. Dekomposisi Deret Waktu & Musiman",
    "3. Dinamika Meteorologi & Arah Angin",
    "4. Klasifikasi AQI & Peta Geospasial",
    "5. Matriks Statistik & Ekspor Data"
])

# ------------------------------------------------------------------------------
# TAB 1: KOMPARASI SPASIAL & ANOMALI STASIUN
# ------------------------------------------------------------------------------
with tab1:
    st.subheader(f"Distribusi Spasial & Deviasi Anomali {selected_pollutant} Antar Stasiun")
    st.caption(
        f"Membandingkan rata-rata dan median konsentrasi {selected_pollutant} terhadap {spec['std_name']} "
        f"serta menghitung persentase deviasi tiap stasiun terhadap rata-rata gabungan seluruh Beijing."
    )

    col_t1_left, col_t1_right = st.columns([1.15, 1.0])

    with col_t1_left:
        st_agg = filtered_df.groupby('station')[selected_pollutant].agg(['mean', 'median']).sort_values(by='mean', ascending=True)
        fig_s1, ax_s1 = plt.subplots(figsize=(8.5, 5.6))

        y_idx = np.arange(len(st_agg))
        bar_cols = [
            '#0F766E' if v == st_agg['mean'].min()
            else ('#B91C1C' if v >= st_agg['mean'].nlargest(min(3, len(st_agg))).min() else '#64748B')
            for v in st_agg['mean'].values
        ]

        ax_s1.barh(y_idx, st_agg['mean'], height=0.68, color=bar_cols, label='Rata-Rata (Mean)')
        ax_s1.scatter(st_agg['median'], y_idx, color='#0F172A', zorder=5, s=45, marker='D', label='Nilai Tengah (Median)')
        ax_s1.axvline(threshold_val, color='#1D4ED8', linestyle='--', linewidth=1.4, label=f'Baku Mutu ({threshold_val:.0f} {unit_str})')

        for i, (m_val, med_val) in enumerate(zip(st_agg['mean'], st_agg['median'])):
            ax_s1.text(m_val + (st_agg['mean'].max() * 0.015), i, f"{m_val:.1f} (Med: {med_val:.1f})", va='center', fontsize=8.5, fontweight='bold')

        ax_s1.set_yticks(y_idx)
        ax_s1.set_yticklabels(st_agg.index, fontweight='bold')
        ax_s1.set_xlim(0, max(st_agg['mean'].max() * 1.25, threshold_val * 1.2))
        ax_s1.set_xlabel(f"Konsentrasi {selected_pollutant} ({unit_str})")
        ax_s1.set_title(f"Peringkat Stasiun Berdasarkan Rata-Rata & Median {selected_pollutant}", fontweight='bold', loc='left', pad=10)
        ax_s1.legend(loc='lower right', frameon=True, fontsize=8.5)
        ax_s1.grid(axis='y')
        plt.tight_layout()
        st.pyplot(fig_s1)

    with col_t1_right:
        city_mean_pol = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)][selected_pollutant].mean()
        city_mean_no2 = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]['NO2'].mean()

        anom_df = filtered_df.groupby('station')[[selected_pollutant, 'NO2']].mean()
        anom_df['pol_anom'] = (anom_df[selected_pollutant] - city_mean_pol) / city_mean_pol * 100
        anom_df['no2_anom'] = (anom_df['NO2'] - city_mean_no2) / city_mean_no2 * 100
        anom_df = anom_df.loc[st_agg.index]

        fig_s2, ax_s2 = plt.subplots(figsize=(7.5, 5.6))
        y_pos = np.arange(len(anom_df))

        ax_s2.barh(y_pos - 0.18, anom_df['pol_anom'], height=0.35, color='#B91C1C', label=f'Deviasi {selected_pollutant} (%)')
        ax_s2.barh(y_pos + 0.18, anom_df['no2_anom'], height=0.35, color='#334155', label='Deviasi NO2 / Indikator Lalu Lintas (%)')
        ax_s2.axvline(0, color='#0F172A', linewidth=1.1)

        ax_s2.set_yticks(y_pos)
        ax_s2.set_yticklabels(anom_df.index, fontweight='bold')
        ax_s2.set_xlabel("Persentase Deviasi terhadap Rata-Rata Kota Beijing (%)")
        ax_s2.set_title("Indeks Anomali Stasiun terhadap Rata-Rata Kota (%)", fontweight='bold', loc='left', pad=10)
        ax_s2.legend(loc='lower right', frameon=True, fontsize=8.5)
        plt.tight_layout()
        st.pyplot(fig_s2)

    st.markdown("#### Matriks Distribusi Statistik Stasiun Pemantauan")
    stat_table = filtered_df.groupby(['station', 'zone'])[selected_pollutant].agg(
        Observasi_Hari='count',
        Rata_Rata='mean',
        Standar_Deviasi='std',
        Minimum='min',
        Kuartil_25=lambda x: np.percentile(x, 25),
        Median='median',
        Kuartil_75=lambda x: np.percentile(x, 75),
        Persentil_95=lambda x: np.percentile(x, 95),
        Maksimum='max',
        Melampaui_Baku_Mutu_Pct=lambda x: (x > threshold_val).mean() * 100
    ).sort_values(by='Rata_Rata', ascending=False).round(2).reset_index()

    st.dataframe(stat_table, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# TAB 2: DEKOMPOSISI DERET WAKTU & MUSIMAN
# ------------------------------------------------------------------------------
with tab2:
    st.subheader(f"Dekomposisi Tren Moving Average ({ma_window} Hari) & Matriks Musiman")
    st.caption(
        "Mengevaluasi tren jangka panjang, disparitas waktu nyata antara Zona Urban dan Suburban, "
        "matriks intensitas Tahun × Bulan, serta oposisi siklus antara partikulat (PM2.5) dan Ozon fotokimia (O3)."
    )

    fig_ts, ax_ts = plt.subplots(figsize=(15, 4.5))

    daily_city = filtered_df.groupby('date')[selected_pollutant].agg(
        mean='mean',
        p25=lambda x: np.percentile(x, 25),
        p75=lambda x: np.percentile(x, 75)
    ).rolling(ma_window, min_periods=3).mean()

    zone_ts = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)].groupby(['date', 'zone'])[selected_pollutant].mean().unstack()
    zone_ts_roll = zone_ts.rolling(ma_window, min_periods=3).mean()

    ax_ts.fill_between(
        daily_city.index, daily_city['p25'], daily_city['p75'],
        color='#CBD5E1', alpha=0.45, label=f'Rentang Interkuartil Stasiun Terpilih (P25–P75, {ma_window}D MA)'
    )
    if 'Urban' in zone_ts_roll.columns:
        ax_ts.plot(zone_ts_roll.index, zone_ts_roll['Urban'], color='#B91C1C', linewidth=2.0, label=f'Zona Urban ({ma_window}-Hari MA)')
    if 'Suburban' in zone_ts_roll.columns:
        ax_ts.plot(zone_ts_roll.index, zone_ts_roll['Suburban'], color='#0F766E', linewidth=2.0, label=f'Zona Suburban ({ma_window}-Hari MA)')

    ax_ts.axhline(threshold_val, color='#1D4ED8', linestyle=':', linewidth=1.3, label=f'Baku Mutu ({threshold_val:.0f} {unit_str})')
    ax_ts.set_title(f"Tren Dekomposisi {ma_window}-Hari Moving Average {selected_pollutant}: Zona Urban vs Suburban", fontweight='bold', loc='left', pad=10)
    ax_ts.set_ylabel(f"{selected_pollutant} ({unit_str})")
    ax_ts.set_xlabel("Tanggal Observasi")
    ax_ts.legend(loc='upper right', frameon=True, ncol=2, fontsize=8.5)
    plt.tight_layout()
    st.pyplot(fig_ts)

    col_t2_left, col_t2_right = st.columns(2)

    with col_t2_left:
        fig_ym, ax_ym = plt.subplots(figsize=(7.5, 4.8))
        ym_piv = filtered_df.pivot_table(index='year', columns='month', values=selected_pollutant, aggfunc='mean')
        ym_piv = ym_piv.reindex(columns=range(1, 13))
        ym_piv.columns = MONTH_NAMES

        sns.heatmap(
            ym_piv, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.5, linecolor='white',
            cbar_kws={'label': f'{selected_pollutant} ({unit_str})'}, ax=ax_ym
        )
        ax_ym.set_title(f"Matriks Intensitas Bulanan {selected_pollutant} (Tahun × Bulan)", fontweight='bold', loc='left', pad=10)
        ax_ym.set_xlabel("Bulan")
        ax_ym.set_ylabel("Tahun")
        plt.tight_layout()
        st.pyplot(fig_ym)

    with col_t2_right:
        fig_seas, ax_seas = plt.subplots(figsize=(7.5, 4.8))
        m_stats = filtered_df.groupby('month')[['PM2.5', 'O3', 'TEMP']].mean().reindex(range(1, 13))

        ax_seas.plot(range(1, 13), m_stats['PM2.5'], marker='o', linewidth=2.2, color='#B91C1C', label='PM2.5 (µg/m³)')
        ax_seas.plot(range(1, 13), m_stats['O3'], marker='s', linewidth=2.2, color='#1D4ED8', label='Ozon / O3 (µg/m³)')

        ax_seas_tw = ax_seas.twinx()
        ax_seas_tw.plot(range(1, 13), m_stats['TEMP'], linestyle='--', marker='^', color='#0F766E', linewidth=1.6, label='Suhu Udara / TEMP (°C)')
        ax_seas_tw.set_ylabel("Suhu Udara Rata-Rata (°C)", color='#0F766E')
        ax_seas_tw.grid(False)

        ax_seas.set_xticks(range(1, 13))
        ax_seas.set_xticklabels(MONTH_NAMES)
        ax_seas.set_xlabel("Bulan Observasi")
        ax_seas.set_ylabel("Konsentrasi Polutan (µg/m³)")
        ax_seas.set_title("Oposisi Siklus Musiman: PM2.5 vs Ozon (O3) & Suhu Udara", fontweight='bold', loc='left', pad=10)

        h1, l1 = ax_seas.get_legend_handles_labels()
        h2, l2 = ax_seas_tw.get_legend_handles_labels()
        ax_seas.legend(h1 + h2, l1 + l2, loc='upper left', frameon=True, fontsize=8.5)
        plt.tight_layout()
        st.pyplot(fig_seas)

    if hourly_df is not None:
        st.markdown("#### Profil Siklus Jam Harian (Diurnal 24-Jam) & Dispersi Kecepatan Angin")
        hr_filt = hourly_df[hourly_df['station'].isin(selected_stations)].groupby('hour')[
            ['PM2.5', 'NO2', 'O3', 'WSPM']
        ].mean().reset_index()

        fig_hr, ax_hr = plt.subplots(figsize=(15, 4.2))
        ax_hr.plot(hr_filt['hour'], hr_filt['PM2.5'], marker='o', linewidth=2.2, color='#B91C1C', label='PM2.5 (µg/m³)')
        ax_hr.plot(hr_filt['hour'], hr_filt['O3'], marker='s', linewidth=2.2, color='#1D4ED8', label='Ozon / O3 (µg/m³)')
        ax_hr.plot(hr_filt['hour'], hr_filt['NO2'], marker='d', linewidth=1.8, linestyle='-.', color='#475569', label='NO2 (µg/m³)')

        ax_hr_tw = ax_hr.twinx()
        ax_hr_tw.plot(hr_filt['hour'], hr_filt['WSPM'], linestyle='--', marker='^', linewidth=2.0, color='#0F766E', label='Kecepatan Angin / WSPM (m/s)')
        ax_hr_tw.set_ylabel("Kecepatan Angin Rata-Rata (m/s)", color='#0F766E')
        ax_hr_tw.grid(False)
        ax_hr_tw.set_ylim(1.0, 2.8)

        ax_hr.set_xticks(range(0, 24))
        ax_hr.set_xticklabels([f"{h:02d}:00" for h in range(0, 24)], rotation=0, fontsize=8.5)
        ax_hr.set_xlabel("Jam Operasional Harian (Waktu Lokal Beijing)")
        ax_hr.set_ylabel("Konsentrasi Polutan (µg/m³)")
        ax_hr.set_title("Dinamika Diurnal 24-Jam: Akumulasi Malam PM2.5, Puncak Fotokimia Sore O3, dan Kecepatan Angin", fontweight='bold', loc='left', pad=10)

        hl1, lb1 = ax_hr.get_legend_handles_labels()
        hl2, lb2 = ax_hr_tw.get_legend_handles_labels()
        ax_hr.legend(hl1 + hl2, lb1 + lb2, loc='upper left', frameon=True, ncol=2, fontsize=8.5)
        plt.tight_layout()
        st.pyplot(fig_hr)

# ------------------------------------------------------------------------------
# TAB 3: DINAMIKA METEOROLOGI & VEKTOR ARAH ANGIN
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Analisis Vektor Arah Mata Angin, Rezim Kecepatan Angin, dan Matriks Korelasi")
    st.caption(
        "Mengidentifikasi arah datangnya transpor polutan regional berdasarkan 16 sektor mata angin "
        "serta mengevaluasi elastisitas konsentrasi polutan terhadap perubahan kecepatan angin dan kelembapan."
    )

    col_t3_left, col_t3_right = st.columns([1.1, 0.9])

    with col_t3_left:
        if 'wd' in filtered_df.columns:
            wd_grp = filtered_df.groupby('wd').agg(
                pol_mean=(selected_pollutant, 'mean'),
                wspm_mean=('WSPM', 'mean'),
                obs_cnt=(selected_pollutant, 'count')
            ).reindex(COMPASS_ORDER).dropna(how='all')

            fig_wd, ax_wd = plt.subplots(figsize=(8.2, 5.2))
            q75_wd = wd_grp['pol_mean'].quantile(0.75)
            q25_wd = wd_grp['pol_mean'].quantile(0.25)
            wd_colors = [
                '#B91C1C' if v >= q75_wd else ('#0F766E' if v <= q25_wd else '#64748B')
                for v in wd_grp['pol_mean']
            ]

            ax_wd.bar(wd_grp.index, wd_grp['pol_mean'], color=wd_colors, width=0.68, label=f'Rata-Rata {selected_pollutant} ({unit_str})')
            ax_wd.axhline(mean_val, color='#0F172A', linestyle='--', linewidth=1.2, label=f'Rata-Rata Periode ({mean_val:.1f} {unit_str})')

            ax_wd_tw = ax_wd.twinx()
            ax_wd_tw.plot(wd_grp.index, wd_grp['wspm_mean'], color='#D97706', marker='o', linewidth=2.0, label='Kecepatan Angin Rata-Rata (m/s)')
            ax_wd_tw.set_ylabel("Kecepatan Angin / WSPM (m/s)", color='#D97706')
            ax_wd_tw.grid(False)

            ax_wd.set_title(f"Profil Konsentrasi {selected_pollutant} & Kecepatan Angin per Sektor Arah Angin", fontweight='bold', loc='left', pad=10)
            ax_wd.set_xlabel("Sektor Arah Mata Angin Dominan (16-Point Compass)")
            ax_wd.set_ylabel(f"Rata-Rata {selected_pollutant} ({unit_str})")
            ax_wd.tick_params(axis='x', rotation=45)

            w1, wl1 = ax_wd.get_legend_handles_labels()
            w2, wl2 = ax_wd_tw.get_legend_handles_labels()
            ax_wd.legend(w1 + w2, wl1 + wl2, loc='upper right', frameon=True, fontsize=8.5)
            plt.tight_layout()
            st.pyplot(fig_wd)
        else:
            st.info("Kolom arah angin ('wd') akan aktif secara otomatis saat data dimuat dari CSV stasiun.")

    with col_t3_right:
        fig_cor, ax_cor = plt.subplots(figsize=(7.2, 5.2))
        corr_vars = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'DEWP', 'PRES', 'RAIN', 'WSPM']
        corr_matrix = filtered_df[corr_vars].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

        sns.heatmap(
            corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            vmin=-1, vmax=1, linewidths=0.5, cbar_kws={'shrink': 0.8}, ax=ax_cor, annot_kws={'size': 8}
        )
        ax_cor.set_title("Matriks Korelasi Pearson Polutan & Parameter Meteorologi", fontweight='bold', loc='left', pad=10)
        plt.tight_layout()
        st.pyplot(fig_cor)

    st.markdown("#### Respons Konsentrasi Polutan terhadap Rezim Kecepatan Angin & Suhu Udara")
    col_reg1, col_reg2 = st.columns(2)

    with col_reg1:
        filtered_df['wind_regime'] = pd.cut(
            filtered_df['WSPM'],
            bins=[-0.1, 1.0, 1.5, 2.0, 2.5, 3.5, 20.0],
            labels=['<1.0 (Stagnan)', '1.0–1.5', '1.5–2.0', '2.0–2.5', '2.5–3.5', '>3.5 (Kencang)']
        )
        w_reg = filtered_df.groupby('wind_regime', observed=False)[['PM2.5', 'O3']].mean()

        fig_wr, ax_wr = plt.subplots(figsize=(7.5, 3.8))
        x_w = np.arange(len(w_reg))
        ax_wr.bar(x_w - 0.18, w_reg['PM2.5'], width=0.35, color='#B91C1C', label='PM2.5 (µg/m³)')
        ax_wr.bar(x_w + 0.18, w_reg['O3'], width=0.35, color='#1D4ED8', label='Ozon / O3 (µg/m³)')
        ax_wr.set_xticks(x_w)
        ax_wr.set_xticklabels(w_reg.index, fontsize=8.5)
        ax_wr.set_xlabel("Rezim Kecepatan Angin Harian (m/s)")
        ax_wr.set_ylabel("Konsentrasi Rata-Rata (µg/m³)")
        ax_wr.set_title("Efek Dispersi Angin terhadap PM2.5 vs Ozon (O3)", fontweight='bold', loc='left', pad=10)
        ax_wr.legend(frameon=True, fontsize=8.5)
        plt.tight_layout()
        st.pyplot(fig_wr)

    with col_reg2:
        filtered_df['temp_regime'] = pd.cut(
            filtered_df['TEMP'],
            bins=[-30, 0, 10, 20, 28, 45],
            labels=['< 0°C (Beku)', '0–10°C (Dingin)', '10–20°C (Sejuk)', '20–28°C (Hangat)', '> 28°C (Panas)']
        )
        t_reg = filtered_df.groupby('temp_regime', observed=False)[['PM2.5', 'SO2', 'O3']].mean()

        fig_tr, ax_tr = plt.subplots(figsize=(7.5, 3.8))
        x_t = np.arange(len(t_reg))
        ax_tr.plot(x_t, t_reg['PM2.5'], marker='o', linewidth=2.2, color='#B91C1C', label='PM2.5 (µg/m³)')
        ax_tr.plot(x_t, t_reg['O3'], marker='s', linewidth=2.2, color='#1D4ED8', label='Ozon / O3 (µg/m³)')
        ax_tr.plot(x_t, t_reg['SO2'], marker='^', linewidth=1.8, linestyle='--', color='#D97706', label='SO2 / Indikator Batu Bara (µg/m³)')
        ax_tr.set_xticks(x_t)
        ax_tr.set_xticklabels(t_reg.index, fontsize=8.5)
        ax_tr.set_xlabel("Rezim Suhu Udara Harian (°C)")
        ax_tr.set_ylabel("Konsentrasi Rata-Rata (µg/m³)")
        ax_tr.set_title("Transisi Emisi Pemanas Musim Dingin vs Fotokimia Musim Panas", fontweight='bold', loc='left', pad=10)
        ax_tr.legend(frameon=True, fontsize=8.5)
        plt.tight_layout()
        st.pyplot(fig_tr)

# ------------------------------------------------------------------------------
# TAB 4: KLASIFIKASI AQI (BINNING) & PETA GEOSPASIAL (ANALISIS LANJUTAN)
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("Analisis Lanjutan: Pengelompokan Risiko Kesehatan (AQI Binning) & Geospasial")
    st.caption(
        "Mengelompokkan observasi harian ke dalam 6 tingkat risiko kesehatan berdasarkan ambang PM2.5 "
        "serta memetakan distribusi spasial stasiun berdasarkan koordinat Lintang dan Bujur."
    )

    col_t4_left, col_t4_right = st.columns([1.05, 0.95])

    with col_t4_left:
        fig_aqi, ax_aqi = plt.subplots(figsize=(8.0, 5.6))
        aqi_ct = pd.crosstab(filtered_df['station'], filtered_df['AQI_Category'], normalize='index') * 100
        aqi_ct = aqi_ct.reindex(columns=AQI_LABELS, fill_value=0)
        order_st = filtered_df.groupby('station')['PM2.5'].mean().sort_values(ascending=True).index
        aqi_ct = aqi_ct.loc[order_st]

        aqi_ct.plot(kind='barh', stacked=True, ax=ax_aqi, color=AQI_COLORS, width=0.72)
        ax_aqi.set_xlim(0, 100)
        ax_aqi.set_xlabel("Proporsi Hari Observasi (%)")
        ax_aqi.set_ylabel("Stasiun Pemantauan")
        ax_aqi.set_title("Distribusi Kategori Kualitas Udara (Binning PM2.5) per Stasiun", fontweight='bold', loc='left', pad=10)
        ax_aqi.legend(title="Kategori AQI (PM2.5 µg/m³)", bbox_to_anchor=(0.5, -0.16), loc='upper center', ncol=2, frameon=True, fontsize=8.5)
        ax_aqi.grid(axis='y')
        plt.tight_layout()
        st.pyplot(fig_aqi)

    with col_t4_right:
        fig_geo, ax_geo = plt.subplots(figsize=(7.5, 5.6))
        geo_df = filtered_df.groupby(['station', 'zone', 'lat', 'lon'])[selected_pollutant].mean().reset_index()

        vmin_g = geo_df[selected_pollutant].min() * 0.95
        vmax_g = geo_df[selected_pollutant].max() * 1.05

        for z_name, grp in geo_df.groupby('zone'):
            marker_style = 'o' if z_name == 'Urban' else '^'
            sc = ax_geo.scatter(
                grp['lon'], grp['lat'],
                s=(grp[selected_pollutant] / geo_df[selected_pollutant].max()) * 420 + 80,
                c=grp[selected_pollutant],
                cmap='YlOrRd', vmin=vmin_g, vmax=vmax_g,
                edgecolors='#0F172A', linewidth=1.2, marker=marker_style, alpha=0.92,
                label=f"Zona {z_name}"
            )

        label_offsets = {
            'Dingling': (0.0, 0.016), 'Changping': (0.0, -0.025), 'Huairou': (0.0, 0.016),
            'Shunyi': (0.0, 0.016), 'Wanliu': (-0.012, 0.016), 'Aotizhongxin': (0.0, 0.016),
            'Gucheng': (0.0, 0.016), 'Guanyuan': (-0.032, 0.008), 'Dongsi': (0.0, 0.015),
            'Nongzhanguan': (0.042, 0.005), 'Wanshouxigong': (-0.038, -0.022), 'Tiantan': (0.035, -0.022)
        }
        for _, row in geo_df.iterrows():
            dx, dy = label_offsets.get(row['station'], (0.0, 0.014))
            ax_geo.text(
                row['lon'] + dx, row['lat'] + dy,
                f"{row['station']} ({row[selected_pollutant]:.1f})",
                fontsize=8, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.8, edgecolor='none')
            )

        cbar = plt.colorbar(sc, ax=ax_geo, pad=0.02)
        cbar.set_label(f"Rata-Rata {selected_pollutant} ({unit_str})", fontsize=9)
        ax_geo.set_title(f"Pemetaan Geospasial Konsentrasi {selected_pollutant} di Beijing", fontweight='bold', loc='left', pad=10)
        ax_geo.set_xlabel("Garis Bujur (Longitude °E)")
        ax_geo.set_ylabel("Garis Lintang (Latitude °N)")
        ax_geo.set_xlim(116.10, 116.75)
        ax_geo.set_ylim(39.82, 40.38)
        ax_geo.legend(loc='lower right', frameon=True, title="Klasifikasi Wilayah")
        plt.tight_layout()
        st.pyplot(fig_geo)

# ------------------------------------------------------------------------------
# TAB 5: MATRIKS STATISTIK & EKSPOR DATA
# ------------------------------------------------------------------------------
with tab5:
    st.subheader("Eksplorasi Rekor Ekstrem & Unduh Dataset Hasil Filter")

    col_top_rec, col_dl = st.columns([3, 1])
    with col_top_rec:
        st.markdown(f"**15 Episode Harian dengan Konsentrasi `{selected_pollutant}` Tertinggi pada Filter Aktif**")
    with col_dl:
        csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Unduh Data Terfilter (CSV)",
            data=csv_bytes,
            file_name=f"beijing_air_quality_{selected_pollutant.lower()}_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    display_cols = ['date', 'station', 'zone', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'TEMP', 'WSPM', 'RAIN', 'AQI_Category']
    if 'wd' in filtered_df.columns:
        display_cols.insert(12, 'wd')

    top_episodes = filtered_df[display_cols].sort_values(by=selected_pollutant, ascending=False).head(15).copy()
    top_episodes['date'] = top_episodes['date'].dt.strftime('%Y-%m-%d')
    st.dataframe(top_episodes.round(2), use_container_width=True, hide_index=True)

    st.markdown("#### Pratinjau Data Harian Teragregasi (100 Baris Pertama)")
    preview_df = filtered_df[display_cols].head(100).copy()
    preview_df['date'] = preview_df['date'].dt.strftime('%Y-%m-%d')
    st.dataframe(preview_df.round(2), use_container_width=True, hide_index=True)
