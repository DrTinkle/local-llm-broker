Unregister-ScheduledTask -TaskName "Local LLM Broker" -Confirm:$false -ErrorAction SilentlyContinue
Write-Host "Local LLM Broker startup registration removed."
