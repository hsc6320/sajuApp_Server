# scripts/clear_gcs.ps1
$bucket = "chatsaju-5cd67-convos"
$object = "conversations.json"

$tmp = New-TemporaryFile
# 빈 JSON으로 초기화 (파일 없음 에러 예방)
Set-Content -Path $tmp -Value "{}" -Encoding UTF8

# Content-Type까지 맞춰서 덮어쓰기
#gsutil -q cp -h "Content-Type:application/json" $tmp "gs://$bucket/$object" | Out-Null
gsutil -q rm -f gs://chatsaju-5cd67-convos/conversations.json

Write-Host "[predeploy] cleared: gs://$bucket/$object"
Remove-Item $tmp -Force
