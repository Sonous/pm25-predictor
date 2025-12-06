# Upload Data to HDFS - PowerShell Script
# Tu dong upload du lieu len HDFS cluster

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "HDFS Data Upload Script for PM2.5 Predictor" -ForegroundColor Cyan
Write-Host "============================================================`n" -ForegroundColor Cyan

# Step 1: Copy data to container
Write-Host "[1/3] Copying data to HDFS container..." -ForegroundColor Yellow
try {
    docker cp data/raw hdfs-namenode:/tmp/raw
    Write-Host "[OK] Data copied successfully" -ForegroundColor Green
} catch {
    Write-Host "[FAIL] Failed to copy data to container" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    exit 1
}

# Step 2: Upload to HDFS
Write-Host "`n[2/3] Uploading files to HDFS..." -ForegroundColor Yellow
try {
    docker exec hdfs-namenode bash -c "hdfs dfs -mkdir -p /data/raw; hdfs dfs -mkdir -p /data/processed; hdfs dfs -mkdir -p /models; hdfs dfs -mkdir -p /results; hdfs dfs -put -f /tmp/raw/*.csv /data/raw/"
    Write-Host "[OK] Files uploaded to HDFS" -ForegroundColor Green
} catch {
    Write-Host "[FAIL] Failed to upload files" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    exit 1
}

# Step 3: Verify upload
Write-Host "`n[3/3] Verifying upload..." -ForegroundColor Yellow
Write-Host "`nFiles in HDFS /data/raw:" -ForegroundColor Cyan
docker exec hdfs-namenode hdfs dfs -ls /data/raw

Write-Host "`nSummary:" -ForegroundColor Cyan
docker exec hdfs-namenode hdfs dfs -count -h /data/raw

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "[SUCCESS] Upload completed successfully!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green

Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "  1. View files in Web UI: http://localhost:9870" -ForegroundColor White
Write-Host "  2. Browse HDFS: Utilities -> Browse the file system -> /data/raw" -ForegroundColor White
Write-Host "  3. Use in notebook: notebooks/01_data_preprocessing_hdfs.ipynb" -ForegroundColor White
Write-Host ""
