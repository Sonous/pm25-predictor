# Notebook: Data Preprocessing with HDFS

## Mô tả

Notebook `01_data_preprocessing_hdfs.ipynb` thực hiện tiền xử lý dữ liệu PM2.5 với PySpark và HDFS. Đây là phiên bản được tối ưu cho môn học Big Data, sử dụng HDFS làm storage layer.

## Tính năng chính

1. **Load dữ liệu từ HDFS** - Đọc raw data từ distributed file system
2. **Xử lý với PySpark** - Spark local mode (không cần Spark cluster)
3. **Save vào HDFS** - Lưu processed data vào HDFS dưới dạng Parquet
4. **Distributed Storage** - Tận dụng HDFS cho Big Data workflow

## Chuẩn bị

### 1. Khởi động HDFS Cluster

```powershell
# Trong thư mục dự án
docker-compose up -d

# Kiểm tra trạng thái
docker-compose ps
```

### 2. Upload Raw Data lên HDFS

```powershell
# Cài đặt dependencies
pip install hdfs pyarrow

# Chạy script upload
python upload_to_hdfs.py
```

### 3. Verify HDFS

- Truy cập NameNode UI: http://localhost:9870
- Browse filesystem: http://localhost:9870/explorer.html#/data/raw
- Kiểm tra có các file CSV trong `/data/raw/`

## Cấu trúc HDFS

```
hdfs://localhost:9000/
└── data/
    ├── raw/                                # Input (upload trước khi chạy)
    │   ├── pollutant_location_7727.csv
    │   ├── pollutant_location_7728.csv
    │   ├── weather_location_7727.csv
    │   ├── weather_location_7728.csv
    │   └── ... (14 locations total)
    │
    └── processed/                          # Output (được tạo bởi notebook)
        ├── cnn_sequences/
        │   ├── train/
        │   ├── val/
        │   └── test/
        ├── lstm_sequences/
        │   ├── train/
        │   ├── val/
        │   └── test/
        ├── xgboost/
        │   ├── train/
        │   ├── val/
        │   └── test/
        ├── scaler_params_json/             # JSON metadata
        └── datasets_ready_json/            # JSON metadata
```

## Chạy Notebook

### Bước 1: Kết nối HDFS

```python
# Cell đầu tiên sẽ setup Spark với HDFS
HDFS_NAMENODE = "hdfs://localhost:9000"
spark = SparkSession.builder \
    .config("spark.hadoop.fs.defaultFS", HDFS_NAMENODE) \
    .master("local[8]") \
    .getOrCreate()
```

### Bước 2: Load từ HDFS

```python
# Đọc dữ liệu từ HDFS
df = spark.read.csv(
    "hdfs://localhost:9000/data/raw/pollutant_location_7727.csv",
    header=True,
    inferSchema=True
)
```

### Bước 3: Xử lý dữ liệu

Notebook thực hiện:

- Data cleaning (outlier removal, missing values)
- Feature engineering (lag features, time features)
- Normalization (Min-Max scaling)
- Train/Val/Test split (70/15/15)
- Sequence creation for Deep Learning models

### Bước 4: Save vào HDFS

```python
# Lưu processed data vào HDFS dưới dạng Parquet
df.write.mode("overwrite") \
    .parquet("hdfs://localhost:9000/data/processed/cnn_sequences/train")
```

## Output Datasets

Sau khi chạy notebook, bạn sẽ có:

### 1. CNN1D-BLSTM Sequences (48 timesteps)

- **Train**: ~200K sequences
- **Val**: ~40K sequences
- **Test**: ~40K sequences
- **Format**: Parquet với array columns

### 2. LSTM Sequences (24 timesteps)

- **Train**: ~200K sequences
- **Val**: ~40K sequences
- **Test**: ~40K sequences
- **Format**: Parquet với array columns

### 3. XGBoost Features (flat)

- **Train**: ~200K records
- **Val**: ~40K records
- **Test**: ~40K records
- **Format**: Parquet với lag features

### 4. Metadata Files

- `scaler_params_json/`: Min-Max scaler parameters
- `datasets_ready_json/`: Dataset metadata và statistics

## Load Processed Data từ HDFS

### Option 1: Load với Spark (Recommended for Big Data)

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .master("local[*]") \
    .getOrCreate()

# Load data
df_train = spark.read.parquet("hdfs://localhost:9000/data/processed/cnn_sequences/train")

# Convert to Pandas nếu cần (cẩn thận với dữ liệu lớn)
pdf_train = df_train.toPandas()
```

### Option 2: Load với Pandas + PyArrow

```python
import pandas as pd

# Pandas có thể đọc trực tiếp từ HDFS với fsspec
df_train = pd.read_parquet(
    "hdfs://localhost:9000/data/processed/cnn_sequences/train"
)
```

### Option 3: Load với hdfs library

```python
from hdfs import InsecureClient
import pandas as pd
import io

client = InsecureClient('http://localhost:9870', user='root')

# Download file
with client.read('/data/processed/xgboost/train/part-00000.parquet') as reader:
    df = pd.read_parquet(io.BytesIO(reader.read()))
```

## Troubleshooting

### HDFS Connection Failed

```
[ERROR] ✗ HDFS connection failed
```

**Giải pháp:**

1. Kiểm tra HDFS đang chạy: `docker-compose ps`
2. Restart cluster: `docker-compose restart`
3. Check logs: `docker-compose logs namenode`

### Data Not Found in HDFS

```
[ERROR] Could not list HDFS files
```

**Giải pháp:**

1. Upload dữ liệu: `python upload_to_hdfs.py`
2. Verify trên UI: http://localhost:9870/explorer.html#/data/raw

### Java/Spark Errors

```
[ERROR] JAVA_HOME not set
```

**Giải pháp:**

```python
import os
os.environ['JAVA_HOME'] = r'C:\Program Files\Java\jdk-21'
```

### Memory Issues

```
[ERROR] OutOfMemoryError
```

**Giải pháp:**

- Giảm `spark.driver.memory` hoặc `spark.executor.memory`
- Process từng location một lần
- Tăng RAM cho Docker Desktop

## Tips & Best Practices

### 1. Performance Optimization

```python
# Cache intermediate results
df = df.cache()
df.count()  # Trigger caching

# Repartition for better parallelism
df = df.repartition(8, "location_id")

# Use Parquet compression
df.write.option("compression", "snappy").parquet(path)
```

### 2. Monitor HDFS Usage

```bash
# Kiểm tra disk usage
docker exec hdfs-namenode hdfs dfs -du -h /data

# Kiểm tra cluster health
docker exec hdfs-namenode hdfs dfsadmin -report
```

### 3. Cleanup Old Data

```bash
# Xóa processed data để chạy lại
docker exec hdfs-namenode hdfs dfs -rm -r /data/processed

# Hoặc xóa toàn bộ
docker-compose down -v  # WARNING: Xóa tất cả data!
```

## So sánh với Local Storage

| Feature             | HDFS                | Local Storage              |
| ------------------- | ------------------- | -------------------------- |
| **Scalability**     | ✅ Unlimited        | ❌ Limited by disk         |
| **Fault Tolerance** | ✅ Replication=2    | ❌ Single point of failure |
| **Distributed**     | ✅ Yes              | ❌ No                      |
| **Speed**           | ⚠️ Network overhead | ✅ Fast (local disk)       |
| **Big Data**        | ✅ Designed for it  | ❌ Not suitable            |
| **Setup**           | ⚠️ Complex          | ✅ Simple                  |

## Next Steps

Sau khi có processed data trên HDFS:

1. **Train Models**: Sử dụng data từ HDFS để train CNN-BiLSTM, LSTM, XGBoost
2. **Distributed Training**: Scale training với Spark MLlib
3. **Production**: Deploy với HDFS làm data lake
4. **Monitoring**: Track data quality và lineage

## Tài liệu tham khảo

- [HDFS Architecture](https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-hdfs/HdfsDesign.html)
- [PySpark + HDFS](https://spark.apache.org/docs/latest/api/python/getting_started/quickstart_df.html)
- [Docker Compose HDFS](https://github.com/big-data-europe/docker-hadoop)
- [Parquet Format](https://parquet.apache.org/docs/)
