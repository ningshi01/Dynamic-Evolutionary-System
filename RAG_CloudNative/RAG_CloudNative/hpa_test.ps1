param(
    [string]$Url,
    [string]$Message = "hello",
    [int]$DelayMs = 100,
    [switch]$Quiet
)

# ����Ҫ����
if (-not $Url) {
    Write-Host "����: ����ָ��API URL" -ForegroundColor Red
    Write-Host "�÷�: .\continuous-api-request.ps1 -Url <api-url> [-Message <message>] [-DelayMs <milliseconds>] [-Quiet]" -ForegroundColor Yellow
    exit 1
}

Write-Host "����API����ű�����" -ForegroundColor Green
Write-Host "Ŀ��: $Url" -ForegroundColor Cyan
Write-Host "��Ϣ: $Message" -ForegroundColor Cyan
Write-Host "���: ${DelayMs}ms" -ForegroundColor Cyan
Write-Host "�� Ctrl+C ֹͣ`n" -ForegroundColor Red

$counter = 0

try {
    while ($true) {
        $counter++
        
        if (-not $Quiet) {
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] ���� #$counter" -ForegroundColor Gray
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
                    Write-Host "   ��Ӧ: $($response.choices[0].message.content)" -ForegroundColor Green
                } else {
                    Write-Host "   ����ɹ�" -ForegroundColor Green
                }
            }
            
        } catch {
            if (-not $Quiet) {
                Write-Host "   ����: $($_.Exception.Message)" -ForegroundColor Red
            }
        }
        
        Start-Sleep -Milliseconds $DelayMs
    }
} finally {
    Write-Host "`n�ű�ֹͣ���ܹ������� $counter ������" -ForegroundColor Yellow
}