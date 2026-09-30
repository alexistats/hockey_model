<#
Register (or remove) the Windows scheduled tasks for the news catch-up.

    powershell -NoProfile -ExecutionPolicy Bypass -File install_tasks.ps1              # install or update
    powershell -NoProfile -ExecutionPolicy Bypass -File install_tasks.ps1 -Uninstall   # remove them all

Three slots a day, local time: morning 06:00, evening 17:00 (before games lock),
night 23:00 (after them). Change a time with e.g. -Evening 17:30 and re-run.

The tasks run as the current user, only while logged on. Each has two triggers:
its daily time, and logging on, so a run missed while the computer was off, asleep
or logged out catches up at the next logon. The runner's -At guard makes the logon
trigger a no-op before the slot's time or once the slot has run today. Logon delays
are staggered (2, 6, 10 minutes) so overdue slots run in order and never fetch at
once. A missed start is also retried as soon as possible. They run on battery too.
#>
param(
    [switch]$Uninstall,
    [string]$Morning = "06:00",
    [string]$Evening = "17:00",
    [string]$Night = "23:00"
)

$runner = Join-Path $PSScriptRoot "run_catchup.ps1"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$slots = @(
    @{ Slot = "morning"; At = $Morning; Delay = "PT2M" },
    @{ Slot = "evening"; At = $Evening; Delay = "PT6M" },
    @{ Slot = "night"; At = $Night; Delay = "PT10M" }
)

# Every task this installer ever made, including slots since renamed or dropped.
Get-ScheduledTask -TaskName "hockey-gm news (*)" -ErrorAction SilentlyContinue |
    ForEach-Object { Unregister-ScheduledTask -TaskName $_.TaskName -Confirm:$false }
if ($Uninstall) { Write-Output "Removed the hockey-gm news tasks."; exit 0 }

foreach ($s in $slots) {
    $name = "hockey-gm news ($($s.Slot))"
    $action = New-ScheduledTaskAction -Execute "powershell.exe" -WorkingDirectory $root `
        -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$runner`" -Slot $($s.Slot) -At $($s.At)"
    $daily = New-ScheduledTaskTrigger -Daily -At $s.At
    $logon = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
    $logon.Delay = $s.Delay
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 30) -MultipleInstances IgnoreNew
    Register-ScheduledTask -TaskName $name -Action $action -Trigger @($daily, $logon) -Settings $settings `
        -Description "hockey_models: gamedaytweets.com catch-up and digest ($($s.Slot)), via .claude/skills/hockey-gm/scheduled/run_catchup.ps1" | Out-Null
    Write-Output "Scheduled '$name' daily at $($s.At)"
}
