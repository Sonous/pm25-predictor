# Data Preprocessing Steps

Tài liệu này mô tả chi tiết các bước tiền xử lý dữ liệu được thực hiện trong `notebooks/01_data_preprocessing.ipynb`.

## 1. Kết nối Spark Cluster
- Khởi tạo `SparkSession` với cấu hình tối ưu cho từng môi trường (Kaggle, Colab, Local).
- Cấu hình bộ nhớ (Driver/Executor memory), parallelism, và Arrow execution để tăng hiệu năng.

## 2. Định nghĩa Schema và Scan Files
- **Schema Definition**:
    - `openaq_schema`: Định nghĩa cấu trúc cho dữ liệu ô nhiễm (location_id, datetime, lat, lon, parameter, value, ...).
    - `weather_schema`: Định nghĩa cấu trúc cho dữ liệu thời tiết (temperature, humidity, wind_speed, ...).
- **Scan & Map Files**:
    - Quét thư mục dữ liệu raw để tìm các file `pollutant_location_*.csv`.
    - Map từng file pollutant với file weather tương ứng dựa trên `location_id`.
- **Initial Processing**:
    - Đọc dữ liệu pollutant, lọc lấy các chỉ số quan trọng: `pm25`, `pm10`, `so2`, `no2`.
    - Đọc dữ liệu weather, loại bỏ missing values.
    - Pivot dữ liệu pollutant để mỗi chất ô nhiễm là một cột.
    - Join dữ liệu pollutant và weather theo `datetime`.
    - Gộp dữ liệu từ tất cả 14 trạm đo (locations) thành một dataset duy nhất (`df_combined`).

## 3. Tổng quan Dataset
- Thống kê số lượng bản ghi theo từng location.
- Kiểm tra khoảng thời gian (Time Range) của từng location.
- Kiểm tra tỷ lệ Missing Values cho từng cột.
- Tính toán thống kê mô tả (mean, std, min, max) cho các đặc trưng.

## 4. Làm sạch Dữ liệu (Data Cleaning)
### 4.1. Loại bỏ Outliers
- Áp dụng các ngưỡng chuẩn quốc tế (WHO/EPA) để loại bỏ giá trị bất thường:
    - **PM2.5 (Target)**: Loại bỏ bản ghi nếu PM2.5 là `null` hoặc ngoài khoảng [0, 250].
    - **PM10**: [0, 430]
    - **NO2**: [0, 400]
    - **SO2**: [0, 500]
    - **Precipitation**: [0, 100]

### 4.2. Xử lý Missing Values (Interpolation)
- Sử dụng **Linear Interpolation** theo thời gian thực (epoch) để điền giá trị thiếu cho các features (`PM10`, `NO2`, `SO2`).
- Chiến lược:
    1.  **Linear Interpolation**: Nội suy dựa trên giá trị trước và sau.
    2.  **Forward Fill**: Điền giá trị trước đó nếu thiếu ở cuối.
    3.  **Backward Fill**: Điền giá trị sau đó nếu thiếu ở đầu.
- **Lưu ý**: Không nội suy chéo giữa các location (partition by `location_id`).
- Xử lý edge cases: Loại bỏ các bản ghi vẫn còn `null` sau khi nội suy (thường là do không có dữ liệu xung quanh).

## 5. Feature Engineering & Normalization
### Bước 1: Time Features
- Tạo các đặc trưng thời gian từ `datetime`:
    - **Cyclic Encoding (Sin/Cos)**: `hour`, `month`, `day_of_week`, `wind_direction`. Giúp mô hình hiểu tính chu kỳ của thời gian.
    - **Binary Feature**: `is_weekend` (1 nếu là T7/CN, 0 nếu là ngày thường).

### Bước 2: Temporal Split (Tránh Data Leakage)
- Chia dữ liệu theo thời gian:
    - **Train**: 70% (từ đầu đến ~11/2024)
    - **Validation**: 15% (tiếp theo đến ~04/2025)
    - **Test**: 15% (còn lại đến 09/2025)

### Bước 3: Normalization
- Sử dụng **Min-Max Scaling** để đưa các đặc trưng dạng số về khoảng [0, 1].
- **Quan trọng**: Chỉ tính toán `min` và `max` từ tập **Train**, sau đó áp dụng cho cả Val và Test để tránh rò rỉ dữ liệu (Data Leakage).
- Các cột được chuẩn hóa: `PM2_5`, `PM10`, `NO2`, `SO2`, `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`, `wind_direction_10m`, `precipitation`.

### Bước 4: Lưu Scaler Parameters
- Lưu các tham số `min`, `max` ra file `scaler_params.json` để sử dụng cho việc denormalize dự đoán sau này.

### Bước 5: Lag Features (Chỉ cho XGBoost)
- Tạo các đặc trưng trễ (Lag features) cho XGBoost:
    - Các bước trễ: 1h, 2h, 3h, 6h, 12h, 24h.
    - Tạo từ các cột **đã chuẩn hóa** (scaled).
- Xử lý `null` sinh ra do tạo lag (24h đầu tiên): Loại bỏ các bản ghi này.

### Bước 6: Chuẩn bị Features cho từng Model
- **Deep Learning (CNN, LSTM)**:
    - Features: 15 đặc trưng (3 pollutants, 5 weather, 6 time cyclic, 1 time binary).
    - Không dùng lag features (mô hình tự học chuỗi).
- **XGBoost**:
    - Features: 63 đặc trưng (15 base features + 48 lag features).

### Bước 7: Tạo Final Datasets
- Tạo các DataFrame riêng biệt cho từng model và từng tập (Train/Val/Test).

### Bước 8: Lưu Metadata
- Lưu thông tin về features, cấu hình lag, và thống kê dataset vào `feature_metadata.json`.

## 6. Sequence Creation (Cho Deep Learning)
- Tạo dữ liệu dạng chuỗi (Sequence) cho các mô hình Deep Learning:
    - **CNN1D-BLSTM**: Sequence length = 48 (48 giờ quá khứ).
    - **LSTM**: Sequence length = 24 (24 giờ quá khứ).
- **Quy trình**:
    1.  **Layer 1**: Loại bỏ N bản ghi đầu tiên của mỗi location (do không đủ lịch sử).
    2.  **Batch Processing**: Tạo sequence theo batch để tối ưu bộ nhớ và tránh lỗi StackOverflow của Spark.
    3.  **Layer 2**: Lọc bỏ bất kỳ sequence nào chứa giá trị `null` (đảm bảo dữ liệu sạch tuyệt đối).

## 7. Export Data
- Xuất dữ liệu đã xử lý ra định dạng **Parquet** (tối ưu cho Spark/Pandas):
    - `processed/cnn_sequences/{train,val,test}`
    - `processed/lstm_sequences/{train,val,test}`
    - `processed/xgboost/{train,val,test}`
- Lưu các file metadata: `datasets_ready.json`, `scaler_params.json`, `feature_metadata.json`.
