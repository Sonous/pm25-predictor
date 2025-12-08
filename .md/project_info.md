## Dataset

- **OpenAQ data info:**

Các cột dữ liệu khi lấy từ openaq: location_id sensors_id location datetime lat lon parameter units value
Với parameter là các chỉ số PM2.5, PM10, NO2, SO2

- **Dữ liệu khí tượng lấy từ OpenMeteo:**

Các cột dữ liệu khi lấy từ OpenMeteo: time temperature_2m relative_humidity_2m wind_speed_10m wind_direction_10m surface_pressure precipitation

# Usage models

- Phương pháp chính: Mô hình đề xuất trong đề tài gồm ba khối chính: CNN1D, BiLSTM và Attention, được kết hợp theo hướng tuần tự để tận dụng khả năng trích xuất đặc trưng cục bộ, học phụ thuộc chuỗi và tập trung vào các yếu tố quan trọng của dữ liệu.
- So sánh với các mô hình học máy khác: LSTM, XGBoost
