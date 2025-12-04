#!/usr/bin/env pwsh
$ports = 2000,2001,5000,5001,8000,3000
foreach($p in $ports){
  try{
    $pattern = ":{0}\s" -f $p
    $lines = netstat -ano | Select-String -Pattern $pattern
    if($lines){
      foreach($line in $lines){
        $parts = ($line -replace '^\s+','') -split '\s+';
        $procId = $parts[-1]
        if($procId -and $procId -match '^[0-9]+$'){
          Write-Host "Killing PID $procId listening on port $p"
          cmd /c "taskkill /PID $procId /F" | Out-Null
        }
      }
    } else {
      Write-Host "No process found on port $p"
    }
  } catch { Write-Host ("Error while checking port {0}: {1}" -f $p, $_) }
}
