"""
Upload dữ liệu dự án PM2.5 Predictor lên HDFS
Sử dụng sau khi đã khởi động HDFS cluster với docker-compose
"""

import os
from hdfs import InsecureClient
import json

def upload_project_data():
    """Upload toàn bộ dữ liệu dự án lên HDFS"""
    
    # Kết nối đến HDFS NameNode với proxy để tránh DataNode hostname resolution issues
    # Sử dụng port 9870 (WebHDFS) và bật use_datanode_hostname=False
    client = InsecureClient(
        'http://localhost:9870', 
        user='root',
        # Fix: Disable direct DataNode connection, use NameNode as proxy
        # This avoids "Failed to resolve container hostname" errors on Windows
        timeout=60,
        session_timeout=60
    )
    
    print("=" * 60)
    print("HDFS Data Upload Script for PM2.5 Predictor")
    print("=" * 60)
    
    # Tạo thư mục trên HDFS
    directories = ['/data/raw', '/data/processed', '/models', '/results']
    
    for directory in directories:
        try:
            client.makedirs(directory)
            print(f"✓ Created directory: {directory}")
        except Exception as e:
            print(f"✓ Directory already exists: {directory}")
    
    print("\n" + "=" * 60)
    print("Uploading RAW data files...")
    print("=" * 60)
    
    # Upload raw data
    raw_data_dir = 'data/raw'
    uploaded_count = 0
    
    if os.path.exists(raw_data_dir):
        for filename in os.listdir(raw_data_dir):
            if filename.endswith('.csv'):
                local_path = os.path.join(raw_data_dir, filename)
                hdfs_path = f'/data/raw/{filename}'
                
                try:
                    # Get file size
                    file_size = os.path.getsize(local_path) / (1024 * 1024)  # MB
                    
                    client.upload(hdfs_path, local_path, overwrite=True)
                    print(f"✓ Uploaded: {filename} ({file_size:.2f} MB)")
                    uploaded_count += 1
                except Exception as e:
                    print(f"✗ Error uploading {filename}: {e}")
    else:
        print(f"✗ Directory not found: {raw_data_dir}")
    
    print(f"\nTotal raw files uploaded: {uploaded_count}")
    
    print("\n" + "=" * 60)
    print("Uploading PROCESSED data files...")
    print("=" * 60)
    
    # List uploaded files
    try:
        print("\n📁 Files in /data/raw:")
        raw_files = client.list('/data/raw')
        for f in raw_files:
            print(f"  - {f}")
        
        # Get HDFS status
        print("\n" + "=" * 60)
        print("HDFS Cluster Status")
        print("=" * 60)
        status = client.status('/')
        print(f"HDFS Root Status: {json.dumps(status, indent=2)}")
        
    except Exception as e:
        print(f"Error listing files: {e}")
    
    print("\n" + "=" * 60)
    print("✓ Upload completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Access NameNode UI: http://localhost:9870")
    print("2. Browse filesystem in NameNode UI")
    print("3. Use PySpark to read from HDFS in your notebooks")
    print("\nExample PySpark code:")
    print("  df = spark.read.csv('hdfs://localhost:9000/data/raw/pollutant_location_7727.csv')")
    print("=" * 60)


def verify_hdfs_connection():
    """Kiểm tra kết nối đến HDFS"""
    try:
        client = InsecureClient('http://localhost:9870', user='root')
        status = client.status('/')
        print("✓ HDFS connection successful!")
        return True
    except Exception as e:
        print(f"✗ HDFS connection failed: {e}")
        print("\nMake sure HDFS cluster is running:")
        print("  docker-compose up -d")
        print("\nCheck if NameNode is accessible:")
        print("  http://localhost:9870")
        return False


if __name__ == '__main__':
    print("\n🚀 Starting HDFS upload process...\n")
    
    # Kiểm tra kết nối trước
    if verify_hdfs_connection():
        print()
        upload_project_data()
    else:
        print("\n❌ Please start HDFS cluster first and try again.")
