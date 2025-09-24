param(
    [string]$Url,
    [string]$Message = "hello",
    [int]$DelayMs = 100,
    [switch]$Quiet
)

# 检查必要参数
if (-not $Url) {
    Write-Host "错误: 必须指定API URL" -ForegroundColor Red
    Write-Host "用法: .\continuous-api-request.ps1 -Url <api-url> [-Message <message>] [-DelayMs <milliseconds>] [-Quiet]" -ForegroundColor Yellow
    exit 1
}

Write-Host "持续API请求脚本启动" -ForegroundColor Green
Write-Host "目标: $Url" -ForegroundColor Cyan
Write-Host "消息: $Message" -ForegroundColor Cyan
Write-Host "间隔: ${DelayMs}ms" -ForegroundColor Cyan
Write-Host "按 Ctrl+C 停止`n" -ForegroundColor Red

$counter = 0

try {
    while ($true) {
        $counter++
        
        if (-not $Quiet) {
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] 请求 #$counter" -ForegroundColor Gray
        }
        
        try {
            $response = Invoke-RestMethod -Uri $Url -Method Post `
                -Headers @{ "Content-Type" = "application/json" } `
                -Body (@{
                    model = "my_knowledge_base"
                    messages = @(
                        @{
                            role = "user"
                            content = $Message
                        }
                    )
                } | ConvertTo-Json)
            
            if (-not $Quiet) {
                if ($response.choices) {
                    Write-Host "   响应: $($response.choices[0].message.content)" -ForegroundColor Green
                } else {
                    Write-Host "   请求成功" -ForegroundColor Green
                }
            }
            
        } catch {
            if (-not $Quiet) {
                Write-Host "   错误: $($_.Exception.Message)" -ForegroundColor Red
            }
        }
        
        Start-Sleep -Milliseconds $DelayMs
    }
} finally {
    Write-Host "`n脚本停止。总共发送了 $counter 个请求。" -ForegroundColor Yellow
}