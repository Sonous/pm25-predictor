# Data Preprocessing Steps

Tài liệu này mô tả chi tiết các bước tiền xử lý dữ liệu được thực hiện trong `notebooks/01_data_preprocessing.ipynb`.

> **Version:** 3.0 - Refactored with STRATIFIED split, Log transformation, và 2-Layer Null Protection

---

## 1. Kết nối Spark Cluster

### 1.1. Environment Detection & Setup

- Tự động phát hiện môi trường: **Kaggle**, **Google Colab**, hoặc **Local**
- Cài đặt Java và PySpark tự động cho Kaggle/Colab
- Local: Sử dụng Java 21 pre-installed

### 1.2. Spark Session Configuration

- Khởi tạo `SparkSession` với cấu hình tối ưu theo từng môi trường:
  - **Kaggle**: 4 cores, 8GB driver memory
  - **Colab**: 2 cores, 2GB driver memory
  - **Local**: 8 cores, 8GB driver memory (AMD Ryzen optimized)
- Cấu hình:
  - Arrow execution enabled (fast Pandas conversion)
  - Adaptive query execution
  - Coalesce partitions optimization
  - Network timeout: 600s (xử lý data lớn)

## 2. Định nghĩa Schema và Load Data

### 2.1. Schema Definition

**OpenAQ Schema (Pollutant Data):**

```python
openaq_schema = StructType([
    StructField("location_id", StringType(), True),
    StructField("sensors_id", StringType(), True),
    StructField("location", StringType(), True),
    StructField("datetime", TimestampType(), True),
    StructField("lat", DoubleType(), True),
    StructField("lon", DoubleType(), True),
    StructField("parameter", StringType(), True),
    StructField("units", StringType(), True),
    StructField("value", DoubleType(), True)
])
```

**Weather Schema:**

```python
weather_schema = StructType([
    StructField("time", TimestampType(), True),
    StructField("temperature_2m", DoubleType(), True),
    StructField("relative_humidity_2m", DoubleType(), True),
    StructField("wind_speed_10m", DoubleType(), True),
    StructField("wind_direction_10m", DoubleType(), True),
    StructField("surface_pressure", DoubleType(), True),
    StructField("precipitation", DoubleType(), True)
])
```

### 2.2. File Scanning & Mapping

- **Adaptive Path Resolution:**
  - Kaggle: `/kaggle/input/{dataset-name}/`
  - Colab: `/content/drive/MyDrive/pm25-data/raw`
  - Local: `../data/raw`
- Quét và map files:
  - Tìm tất cả `pollutant_location_*.csv`
  - Map với `weather_location_*.csv` tương ứng
  - Verify cả 2 files tồn tại cho mỗi location

### 2.3. Per-Location Processing

Xử lý từng location riêng biệt, sau đó union:

**Bước 1: Load Pollutant Data**

- Đọc CSV với schema defined
- Filter chỉ lấy 4 pollutants: `pm25`, `pm10`, `so2`, `no2`
- Pivot từ long format → wide format (mỗi pollutant = 1 cột)

**Bước 2: Load Weather Data**

- Đọc CSV với schema defined
- Rename column `time` → `datetime` (để join)
- Drop missing values

**Bước 3: Join Pollutant + Weather**

- Inner join theo `datetime`
- Chỉ giữ records có đầy đủ cả air quality và weather data

**Bước 4: Union All Locations**

- Gộp dữ liệu từ 14 locations thành một DataFrame duy nhất
- Total: ~300,000 records (14 locations × ~21,000 records/location)

## 3. Exploratory Data Analysis

### 3.1. Dataset Statistics

- **Total Records:** ~300,000
- **Time Range:** 2023-01 → 2025-09 (~2.5 years)
- **Locations:** 14 monitoring stations
- **Features:** 12 columns (4 pollutants + 7 weather + 1 datetime)

### 3.2. Per-Location Analysis

```python
df_combined.groupBy("location_id").count().show()
```

- Mỗi location: ~21,000 records
- Time range: Xác minh temporal coverage cho mỗi location

### 3.3. Missing Value Analysis

```python
for col in df_combined.columns:
    null_count = df_combined.filter(F.col(col).isNull()).count()
    null_pct = (null_count / total_count) * 100
```

- **Before Cleaning:**
  - PM2.5: ~5% missing
  - PM10, NO2, SO2: ~8% missing
  - Weather: <1% missing

### 3.4. Descriptive Statistics

```python
df_combined.select([F.mean(c), F.stddev(c), F.min(c), F.max(c)
                    for c in numerical_cols]).show()
```

- Xác định range, outliers tiềm năng
- Detect data quality issues

## 4. Làm sạch Dữ liệu (Data Cleaning)

### 4.1. Loại bỏ Outliers - WHO/EPA Standards

Áp dụng các ngưỡng chuẩn quốc tế để loại bỏ giá trị bất thường:

**Target Variable - PM2.5:**

```python
# PM2.5 là TARGET - PHẢI có giá trị thật
(F.col("PM2_5").isNotNull()) &
(F.col("PM2_5") >= 0) &
(F.col("PM2_5") < 250)  # WHO Emergency threshold
```

- **Ngưỡng:** [0, 250) μg/m³
- **Lý do:** PM2.5 là biến cần dự đoán (target), không được impute
- **Cơ sở:** WHO Emergency threshold 250 μg/m³

**Feature Variables - Pollutants:**

```python
# Features: Cho phép null, chỉ loại outliers
((F.col("PM10").isNull()) | ((F.col("PM10") >= 0) & (F.col("PM10") < 430)))  # WHO Emergency
((F.col("NO2").isNull()) | ((F.col("NO2") >= 0) & (F.col("NO2") < 400)))     # WHO/EU 1-hour
((F.col("SO2").isNull()) | ((F.col("SO2") >= 0) & (F.col("SO2") < 500)))     # WHO/EU 10-min
```

**Weather Features:**

```python
# Precipitation: Phải có giá trị hợp lệ
(F.col("precipitation") >= 0) & (F.col("precipitation") < 100)  # WMO standards
```

**Kết quả sau Outlier Removal:**

- PM2.5: 0 nulls (100% records có giá trị thật)
- PM10, NO2, SO2: Còn missing ~5-10%
- Precipitation: Clean

### 4.2. Log Transformation (PM2.5 Target)

**Vấn đề:** PM2.5 có phân phối lệch (right-skewed)

**Giải pháp:** Áp dụng log transformation

```python
df_cleaned = df_no_outliers.withColumn(
    "PM2_5",
    F.log1p(F.col("PM2_5"))  # log(1 + x) để tránh log(0)
)
```

**Lợi ích:**

- Giảm skewness của phân phối
- Model training ổn định hơn
- Predictions chính xác hơn

**Lưu ý:** Phải inverse transform khi dự đoán: `np.expm1(prediction)`

### 4.3. Xử lý Missing Values - Linear Interpolation

**Chiến lược:** Chỉ impute features, KHÔNG impute target (PM2.5)

**Columns cần impute:**

```python
pollutant_cols = ["PM10", "NO2", "SO2"]  # ⚠️ KHÔNG bao gồm PM2.5!
```

**Phương pháp: True Linear Interpolation (Time-based)**

```python
# Tính toán:
y = y₁ + (y₂ - y₁) × (t - t₁) / (t₂ - t₁)

# Implementation:
interpolated_value = (
    F.col(f"{col_name}_prev_value") +
    (F.col(f"{col_name}_next_value") - F.col(f"{col_name}_prev_value")) *
    ((F.col("epoch") - F.col(f"{col_name}_prev_time")) /
     (F.col(f"{col_name}_next_time") - F.col(f"{col_name}_prev_time")))
)
```

**Fallback Logic:**

1. **Giữ nguyên** - Nếu giá trị đã tồn tại
2. **Linear Interpolation** - Nếu có cả giá trị trước & sau
3. **Forward Fill** - Nếu chỉ có giá trị trước
4. **Backward Fill** - Nếu chỉ có giá trị sau
5. **Drop** - Nếu không có giá trị nào xung quanh

**Ư u điểm:**

- Chính xác về mặt thời gian (sử dụng epoch)
- An toàn cho multi-location data (partition by location_id)
- Scalable với PySpark (không cần convert Pandas)

**Kết quả sau Interpolation:**

- PM2.5: 0 nulls ✅ (Target - không impute)
- PM10: 0 nulls ✅ (Interpolated)
- NO2: 0 nulls ✅ (Interpolated)
- SO2: 0 nulls ✅ (Interpolated)

## 5. Feature Engineering & Normalization

> **Version:** 3.0 - STRATIFIED Split + Log Transform + 2-Layer Protection

### 5.1. Time Features

- Tạo các đặc trưng thời gian từ `datetime`
- **Cyclic Encoding (Sin/Cos):** hour, month, day_of_week (captures circular nature)
- **Binary Feature:** is_weekend (1 nếu T7/CN, 0 nếu ngày thường)
- **Total:** 7 time features (6 cyclic + 1 binary)

### 5.2. STRATIFIED Temporal Split

**CRITICAL:** Split TRƯỚC normalization để tránh data leakage!

- **Chiến lược:** STRATIFIED per month (70/15/15 cho từng tháng)
- **Lợi ích:** Mỗi split có đầy đủ các tháng, tránh seasonal bias
- **Implementation:** Window function partition by (year, month, location_id)

### 5.3. Normalization

- **Min-Max Scaling [0, 1]** cho BASE features only
- **QUAN TRỌNG:** Chỉ tính min/max từ TRAIN set, apply cho val/test
- Columns: PM2_5_log, PM10, NO2, SO2, temperature, humidity, wind, precipitation
- Format: `{column}_scaled`

### 5.4. Lưu Scaler Parameters

- Lưu min/max values vào `scaler_params.json`
- Bao gồm metadata về log transformation
- Sử dụng cho denormalization + inverse log transform khi inference

### 5.5. Lag Features (Chỉ XGBoost)

- **Tạo từ các cột ĐÃ normalize** (preserves scale relationship)
- Lags: 1h, 2h, 3h, 6h, 12h, 24h
- Base columns: 8 scaled features
- Total: 8 × 6 = 48 lag features
- Drop 24h đầu (incomplete history)

### 5.6. Model-specific Features

- **Deep Learning (CNN, LSTM):** 15 features (3 pollutants + 5 weather + 6 cyclic + 1 binary)
  - Không dùng lag features (model tự học từ sequences)
- **XGBoost:** 63 features (15 base + 48 lags)
  - Cần lag features vì không xử lý sequences

### 5.7. Final Datasets

- **dl_train/val/test:** DataFrame cho Deep Learning (15 features)
- **xgb_train/val/test:** DataFrame cho XGBoost (63 features)
- Target: `PM2_5_log_scaled` (log-transformed + normalized)

## 6. Sequence Creation (Deep Learning)

### 6.1. Sequence Generation với 2-Layer Null Protection

**Mục đích:** Tạo chuỗi thời gian cho CNN và LSTM với đảm bảo 100% clean data

**Sequence Lengths:**

- **CNN1D-BLSTM:** 48 timesteps (48 giờ quá khứ)
- **LSTM:** 24 timesteps (24 giờ quá khứ)

**Quy trình:**

1. **Layer 1 - Incomplete History Protection:**

   - Drop N records đầu tiên của mỗi location
   - Đảm bảo mỗi record có đủ lịch sử để tạo sequence

2. **Batch Processing:**

   - Chia features thành batches nhỏ (4-5 features/batch)
   - Tránh StackOverflowError của Spark
   - Checkpoint sau mỗi batch

3. **Layer 2 - Data Gap Protection:**
   - Filter sequences chứa NULL/NaN
   - Sử dụng `forall()` để check từng element
   - Đảm bảo 100% clean sequences

**Kết quả:**

- ~200K CNN sequences (train)
- ~200K LSTM sequences (train)
- ZERO nulls guarantee

### 6.2. Implementation Details

```python
def create_sequences_optimized(df, feature_cols, target_col, sequence_length):
    # Layer 1: Drop incomplete history
    # Layer 2: Filter NULL in sequences
    # Result: 100% clean data
```

## 7. Export Data

### 7.1. Adaptive Output Path

**Multi-Environment Support:**

- **Kaggle:** `/kaggle/working/processed` (auto-saved on commit)
- **Colab:** `/content/drive/MyDrive/pm25-data/processed`
- **Local:** `../data/processed`

### 7.2. Export Format: Parquet

**Datasets được export:**

1. **CNN Sequences:**

   - `processed/cnn_sequences/{train,val,test}/`
   - Format: Parquet với array columns
   - ~200K train / ~40K val / ~40K test

2. **LSTM Sequences:**

   - `processed/lstm_sequences/{train,val,test}/`
   - Format: Parquet với array columns
   - ~200K train / ~40K val / ~40K test

3. **XGBoost Data:**
   - `processed/xgboost/{train,val,test}/`
   - Format: Parquet flat features
   - ~200K train / ~40K val / ~40K test

### 7.3. Metadata Files

**Files được lưu:**

1. **scaler_params.json:**

   - Min/max values cho mỗi feature
   - Log transformation metadata
   - Inverse transform order

2. **feature_metadata.json:**

   - Feature lists (DL vs XGBoost)
   - Lag configuration
   - Dataset counts
   - Pipeline version

3. **datasets_ready.json:**
   - Export timestamp
   - Model configurations
   - Record counts
   - Data format info

## 8. Summary

### Pipeline Execution Order ✅

```
1. Load Data (per-location) → Union all locations
2. EDA → Statistics & Missing analysis
3. Outlier Removal (WHO/EPA standards)
4. Log Transform (PM2.5 only)
5. Interpolation (features only, NOT target)
6. Time Features (cyclic encoding)
7. STRATIFIED Split (70/15/15 per month)
8. Normalization (BASE features, train stats only)
9. Lag Features (FROM SCALED, XGBoost only)
10. Model-specific Datasets
11. Sequence Creation (2-layer protection)
12. Export to Parquet (multi-environment)
```

### Data Quality Guarantees ✅

- ✅ PM2.5 target: 100% real values (no imputation)
- ✅ Features: 0% missing (linear interpolation)
- ✅ Outliers removed (WHO/EPA standards)
- ✅ No data leakage (split before normalization)
- ✅ Scale consistency (lags from scaled columns)
- ✅ Sequences: ZERO nulls (2-layer protection)
- ✅ Log transform applied & documented

### Output Statistics

- **Total Records:** ~300,000 (cleaned from outliers)
- **Locations:** 14 monitoring stations
- **Time Range:** 2023-01 → 2025-09
- **Final Datasets:**
  - Train: ~200,000 records (70%)
  - Val: ~40,000 records (15%)
  - Test: ~40,000 records (15%)
- **Features:**
  - Deep Learning: 15 features
  - XGBoost: 63 features (15 + 48 lags)
- **Target:** PM2_5_log_scaled [0, 1]

---

_Last updated: December 6, 2025 - Version 3.0_
