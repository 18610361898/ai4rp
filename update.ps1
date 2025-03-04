$count = 0
do {
    $count++
    Write-Host "try to do $count-th pushing ..."
    git push -u origin main
    if ($LASTEXITCODE -eq 0) {
        Write-Host "success!"
        break
    }
    Write-Host "failed, try again after 5s ..."
    Start-Sleep -Seconds 5
} while ($true)