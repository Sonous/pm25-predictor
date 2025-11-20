# Mô tả bài toán

## Dataset

- OpenAQ data info:

  - Các cột dữ liệu khi lấy từ openaq: location_id sensors_id location datetime lat lon parameter units value
  - Với parameter là các chỉ số PM2.5, PM10, NO2, SO2
  - Dữ liệu được lấy phân tán trên 14 trạm trên khu vực Hong Kong. Với khoảng thời gian từ 11/2022 - 9/2025

```csv
location_id,sensors_id,location,datetime,lat,lon,parameter,units,value
7727,22558,Tung Chung-7727,2022-11-01T01:00:00+08:00,22.2888888888889,113.943611111111,pm10,µg/m³,100.3
7727,22558,Tung Chung-7727,2022-11-01T02:00:00+08:00,22.2888888888889,113.943611111111,pm10,µg/m³,56.9
7727,22558,Tung Chung-7727,2022-11-01T03:00:00+08:00,22.2888888888889,113.943611111111,pm10,µg/m³,45.1
7727,22558,Tung Chung-7727,2022-11-01T04:00:00+08:00,22.2888888888889,113.943611111111,pm10,µg/m³,46.0
7727,22558,Tung Chung-7727,2022-11-01T05:00:00+08:00,22.2888888888889,113.943611111111,pm10,µg/m³,51.5
7727,22558,Tung Chung-7727,2022-11-01T06:00:00+08:00,22.2888888888889,113.943611111111,pm10,µg/m³,45.7
```

- Dữ liệu khí tượng lấy từ OpenMeteo:

  - Các cột dữ liệu khi lấy từ OpenMeteo: time temperature_2m relative_humidity_2m wind_speed_10m wind_direction_10m surface_pressure precipitation
  - Dựa trên tọa độ của 14 trạm quan trắc trên để lấy dữ liệu thời tiết phù hợp.
  - {'time': 'iso8601',
    'temperature_2m': '°C',
    'relative_humidity_2m': '%',
    'wind_speed_10m': 'km/h',
    'wind_direction_10m': '°',
    'surface_pressure': 'hPa',
    'precipitation': 'mm'},
  - Dữ liệu mẫu:

```csv
time,temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure,precipitation
2022-11-01T00:00,24.5,46,31.7,8,1007.8,0.0
2022-11-01T01:00,23.9,47,33.0,8,1007.4,0.0
2022-11-01T02:00,23.5,49,32.0,8,1006.5,0.0
2022-11-01T03:00,23.3,49,33.3,10,1005.9,0.0
2022-11-01T04:00,23.1,50,34.3,10,1005.7,0.0
2022-11-01T05:00,22.9,50,34.3,9,1005.6,0.0
2022-11-01T06:00,22.7,50,34.2,8,1006.2,0.0
2022-11-01T07:00,22.5,51,33.1,8,1006.8,0.0
2022-11-01T08:00,23.2,49,28.6,11,1008.3,0.0
2022-11-01T09:00,23.2,49,34.7,10,1008.9,0.0
2022-11-01T10:00,23.1,49,32.6,10,1008.9,0.0
2022-11-01T11:00,23.3,49,32.8,3,1008.5,0.0
2022-11-01T12:00,23.5,50,34.5,11,1008.0,0.0
```

- Ứng với mỗi location thì sẽ gồm 2 file chính theo định dạng: pollutant*loction*[location_id].csv và weather*location*[location_id].csv

## Mô hình được sử dụng

- Phương pháp chính: Mô hình đề xuất trong đề tài gồm ba khối chính: CNN1D -> BiLSTM -> Attention
- Các mô hình khác được sử dụng để so sánh với mô hình chính: LSTM, XGBoost

## Mục tiêu bài toán

- Sử dụng các công nghệ big data như HDFS, Spark, Kafka, v.v để xử lý bài toán.
- Xây dựng mô hình chính để dự đoán chỉ số PM2.5
- Sử dụng các mô hình phụ để phục vụ cho yêu cầu so sánh tính chính xác cao của mô hình chính thông qua các chỉ số đo lường như R², RMSE, MRE và MAE.
- Trực quan hóa các chỉ số, dữ liệu.
