#!/bin/bash
# Upload data to HDFS from inside Docker container
# This avoids hostname resolution issues
# 
# USAGE FROM WINDOWS:
#   docker exec -i hdfs-namenode bash < upload_to_hdfs_docker.sh
#
# OR:
#   docker cp upload_to_hdfs_docker.sh hdfs-namenode:/tmp/
#   docker exec hdfs-namenode bash /tmp/upload_to_hdfs_docker.sh

echo "============================================================"
echo "HDFS Data Upload (Docker Internal)"
echo "============================================================"

# Wait for HDFS to be ready
echo "Waiting for HDFS to be ready..."
sleep 5

# Create directories
echo "Creating HDFS directories..."
hdfs dfs -mkdir -p /data/raw
hdfs dfs -mkdir -p /data/processed
hdfs dfs -mkdir -p /models
hdfs dfs -mkdir -p /results

echo "✓ Directories created"

# Check if data is in /tmp/raw (uploaded via docker cp)
if [ -d "/tmp/raw" ]; then
    echo ""
    echo "============================================================"
    echo "Uploading RAW data files from /tmp/raw..."
    echo "============================================================"
    
    # Upload all CSV files
    for file in /tmp/raw/*.csv; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            echo "Uploading: $filename"
            hdfs dfs -put -f "$file" /data/raw/
            echo "✓ Uploaded: $filename"
        fi
    done
    
    echo ""
    echo "============================================================"
    echo "Upload completed!"
    echo "============================================================"
    
    # List uploaded files
    echo ""
    echo "Files in HDFS /data/raw:"
    hdfs dfs -ls /data/raw/
    
    # Show summary
    echo ""
    echo "Summary:"
    hdfs dfs -count -h /data/raw
    
else
    echo "ERROR: /tmp/raw directory not found"
    echo "Please run: docker cp data/raw hdfs-namenode:/tmp/raw"
    exit 1
fi
    echo "Make sure to mount the project directory when running this script"
    exit 1
fi
