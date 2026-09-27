param(
    [string]$Intervention = 'D:\300_clip_nurec\00_raw\ar1_output\reasoning_intervention_nurec_selected_300.jsonl',
    [string]$GroundTruth = 'D:\300_clip_nurec\00_raw\ground_truth\ego_future_gt_nurec_300.jsonl',
    [string]$OutputDir = 'D:\300_clip_nurec\04_analysis\ade',
    [double]$NoChangeTolerance = 0.001
)

$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

function Read-JsonlMap([string]$Path) {
    $map = @{}
    $reader = [System.IO.StreamReader]::new($Path)
    try {
        while (($line = $reader.ReadLine()) -ne $null) {
            if (-not [string]::IsNullOrWhiteSpace($line)) {
                $row = $line | ConvertFrom-Json
                $map[[string]$row.clip_id] = $row
            }
        }
    } finally { $reader.Dispose() }
    return $map
}

function Get-Ade($prediction, $groundTruth) {
    if ($prediction.Count -ne $groundTruth.Count) {
        throw "Waypoint count mismatch: prediction=$($prediction.Count), gt=$($groundTruth.Count)"
    }
    $sum = 0.0
    for ($i = 0; $i -lt $prediction.Count; $i++) {
        $dx = [double]$prediction[$i].x_m - [double]$groundTruth[$i][0]
        $dy = [double]$prediction[$i].y_m - [double]$groundTruth[$i][1]
        $sum += [math]::Sqrt($dx*$dx + $dy*$dy)
    }
    return $sum / $prediction.Count
}

$gt = Read-JsonlMap $GroundTruth
$groups = @{}
$reader = [System.IO.StreamReader]::new($Intervention)
try {
    while (($line = $reader.ReadLine()) -ne $null) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $r = $line | ConvertFrom-Json
        $clip = [string]$r.clip_id
        if (-not $gt.ContainsKey($clip)) { throw "Missing ground truth for $clip" }

        $ground = $gt[$clip].ego_future_xyz
        $baseline = Get-Ade $r.clean_waypoints $ground
        $output = Get-Ade $r.guided_waypoints $ground
        $key = "{0}|{1}" -f $r.mode, ([double]$r.alpha).ToString('0.###',[Globalization.CultureInfo]::InvariantCulture)
        if (-not $groups.ContainsKey($key)) { $groups[$key] = [System.Collections.Generic.List[object]]::new() }
        $groups[$key].Add([pscustomobject]@{
            clip_id=$clip; mode=[string]$r.mode; alpha=[double]$r.alpha
            baseline_ade=$baseline; output_ade=$output; delta_ade=($output-$baseline)
        })
    }
} finally { $reader.Dispose() }

$rows = foreach ($key in ($groups.Keys | Sort-Object)) {
    $items = @($groups[$key])
    $n = $items.Count
    $improve = @($items | Where-Object { $_.delta_ade -lt -$NoChangeTolerance }).Count
    $decrease = @($items | Where-Object { $_.delta_ade -gt $NoChangeTolerance }).Count
    $unchanged = $n - $improve - $decrease
    $sumImprove = (($items | Where-Object { $_.delta_ade -lt -$NoChangeTolerance } | Measure-Object delta_ade -Sum).Sum) * -1
    $sumDecrease = (($items | Where-Object { $_.delta_ade -gt $NoChangeTolerance } | Measure-Object delta_ade -Sum).Sum)
    [pscustomobject]@{
        mode=$items[0].mode; alpha=$items[0].alpha; N=$n
        mean_baseline_ADE=[math]::Round((($items | Measure-Object baseline_ade -Average).Average),4)
        mean_output_ADE=[math]::Round((($items | Measure-Object output_ade -Average).Average),4)
        delta_ADE=[math]::Round((($items | Measure-Object delta_ade -Average).Average),4)
        ADE_decrease_n=$improve; ADE_increase_n=$decrease; no_change_n=$unchanged
        ADE_decrease_pct=[math]::Round(100*$improve/$n,2)
        ADE_increase_pct=[math]::Round(100*$decrease/$n,2)
        no_change_pct=[math]::Round(100*$unchanged/$n,2)
        total_ADE_decrease=[math]::Round($sumImprove,4)
        total_ADE_increase=[math]::Round($sumDecrease,4)
        net_improvement=[math]::Round(($sumImprove-$sumDecrease),4)
    }
}

$csv = Join-Path $OutputDir 'ade_by_mode_alpha.csv'
$json = Join-Path $OutputDir 'ade_by_mode_alpha.json'
$md = Join-Path $OutputDir 'ade_by_mode_alpha.md'
$rows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $csv
$rows | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 -Path $json
$header = '| Mode | Alpha | N clip | Mean baseline ADE | Mean output ADE | ΔADE | ADE giảm | ADE tăng | Không đổi | Tổng ADE giảm | Tổng ADE tăng | Net cải thiện |'
$separator = '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|'
$lines = @($header, $separator)
foreach ($row in $rows) {
    $lines += ('| {0} | {1} | {2} | {3} | {4} | {5} | {6}% | {7}% | {8}% | {9} | {10} | {11} |' -f `
        $row.mode, $row.alpha, $row.N, $row.mean_baseline_ADE, $row.mean_output_ADE,
        $row.delta_ADE, $row.ADE_decrease_pct, $row.ADE_increase_pct, $row.no_change_pct,
        $row.total_ADE_decrease, $row.total_ADE_increase, $row.net_improvement)
}
$lines | Set-Content -Encoding UTF8 -Path $md
Write-Output "Saved: $csv"
Write-Output "Saved: $json"
$rowCount = $rows.Count
Write-Output "Saved: $md ($rowCount rows including alpha=0 baselines)"
$rows | Format-Table mode,alpha,N,mean_baseline_ADE,mean_output_ADE,delta_ADE,ADE_decrease_pct,ADE_increase_pct,no_change_pct,net_improvement -AutoSize
