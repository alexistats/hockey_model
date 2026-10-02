<#
The scheduled news catch-up: what Windows Task Scheduler runs, three times a day.

    powershell -NoProfile -ExecutionPolicy Bypass -File run_catchup.ps1 -Slot morning

1. The fetch runs here, deterministically: news.py reads gamedaytweets.com from the
   bookmark, archives the new tweets and moves the bookmark.
2. A fresh Claude Code session (claude -p) reads that output with the hockey-gm
   skill, the week's plan and the decision log, and replies with a digest. It can
   read, look things up and run gm.py; it cannot edit anything (catchup_settings.json),
   and it starts with no connectors or MCP servers (--strict-mcp-config).
3. The digest is saved to artifacts/season/news/digests/ and announced with a desktop
   notification. The session is named hockey-gm-<slot>-<stamp>, so it can be
   resumed later (claude --resume) to talk it through.

Every run appends a line to artifacts/season/news/runs.log.

The same action serves both of a task's triggers, its daily time and logging on:
with -At, a run happens only once that time has passed today and there is no digest
for the slot from today at or after it. A logon before the time waits for the daily
trigger; a logon after a missed run catches it up; a second trigger that day is a no-op.
-Force runs regardless (by hand).
#>
param(
    [ValidateSet("morning", "evening", "night")][string]$Slot = "morning",
    [string]$At = "",
    [switch]$Force
)

$ErrorActionPreference = "Continue"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = New-Object System.Text.UTF8Encoding $false
# `*>` below goes through Out-File, which writes UTF-16 in Windows PowerShell 5.1,
# and the session's Read tool then sees every character spaced out.
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'

$stamp = Get-Date -Format "yyyy-MM-dd_HHmm"
$newsDir = Join-Path $root "artifacts\season\news"
$digests = Join-Path $newsDir "digests"
New-Item -ItemType Directory -Force $digests | Out-Null
$log = Join-Path $newsDir "runs.log"
function Write-Log($message) {
    Add-Content -Path $log -Encoding utf8 -Value ("{0} [{1}] {2}" -f (Get-Date -Format s), $Slot, $message)
}

if ($At -and -not $Force) {
    $today = Get-Date -Format "yyyy-MM-dd"
    $due = [datetime]::ParseExact("$today $At", "yyyy-MM-dd HH:mm", $null)
    if ((Get-Date) -lt $due) { exit 0 }  # before its time: the daily trigger runs it
    $cutoff = $At -replace ":", ""
    $done = Get-ChildItem $digests -Filter "${today}_*-$Slot.md" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name.Substring(11, 4) -ge $cutoff }
    if ($done) { exit 0 }  # already ran for today's slot
}

function Show-Toast($title, $body, $path) {
    [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
    [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
    $e = { param($s) [System.Security.SecurityElement]::Escape($s) }
    $uri = "file:///" + ($path -replace '\\', '/')
    $xml = "<toast activationType=`"protocol`" launch=`"$(& $e $uri)`"><visual><binding template=`"ToastGeneric`">" +
           "<text>$(& $e $title)</text><text>$(& $e $body)</text></binding></visual></toast>"
    $doc = New-Object Windows.Data.Xml.Dom.XmlDocument
    $doc.LoadXml($xml)
    $app = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe'
    [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($app).Show(
        [Windows.UI.Notifications.ToastNotification]::new($doc))
}

# 1. The fetch.
$news = Join-Path $digests "$stamp-$Slot.news.txt"
& (Join-Path $root ".venv\Scripts\python.exe") ".claude\skills\hockey-gm\news.py" --pages 15 *> $news
if ($LASTEXITCODE -ne 0) {
    Write-Log "news.py failed (exit $LASTEXITCODE); see $news"
    try { Show-Toast "Hockey news ($Slot): fetch failed" "See runs.log" $news } catch {}
    exit 1
}
$counts = Get-Content $news -TotalCount 1
Write-Log "fetched: $counts"

# 2. The read, in a fresh session.
$claude = Join-Path $env:APPDATA "npm\node_modules\@anthropic-ai\claude-code\bin\claude.exe"
if (-not (Test-Path $claude)) { $claude = (Get-Command claude -ErrorAction SilentlyContinue).Source }
$focus = switch ($Slot) {
    "morning" { "This is the early-morning run: lead with overnight injuries and transactions, and what to watch today (which of my players play tonight, and any open question the day's practices will settle)." }
    "evening" { "This is the evening run: tonight's games lock soon, so lead with confirmed starting goalies, morning-skate line and power-play changes, and late scratches." }
    "night" { "This is the night run, after tonight's games: lead with in-game injuries, post-game news and lineup changes, and anything that changes tomorrow's lineup or the week's adds." }
}
$relNews = $news.Substring($root.Length + 1) -replace '\\', '/'
$prompt = (Get-Content (Join-Path $PSScriptRoot "catchup_prompt.md") -Raw -Encoding utf8).
    Replace("{SLOT}", $Slot).Replace("{STAMP}", $stamp).Replace("{NEWS}", $relNews).Replace("{SLOT_FOCUS}", $focus)
$digest = Join-Path $digests "$stamp-$Slot.md"
$settings = Join-Path $PSScriptRoot "catchup_settings.json"
$output = $prompt | & $claude -p --settings $settings --permission-mode dontAsk --strict-mcp-config --name "hockey-gm-$Slot-$stamp" --output-format text 2>&1
$code = $LASTEXITCODE
$output | Out-File -FilePath $digest -Encoding utf8
Write-Log "digest: $digest (claude exit $code)"

# 3. Tell the user it's there.
$headline = (Get-Content $digest -TotalCount 1 -Encoding utf8) -replace '^[#\s*]+', ''
if (-not $headline) { $headline = "Digest written; see the file" }
try { Show-Toast "Hockey news ($Slot)" $headline $digest } catch { Write-Log "notification failed: $_" }
exit $code
