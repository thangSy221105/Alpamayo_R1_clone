param(
    [string]$Intervention = 'D:\300_clip_nurec\00_raw\ar1_output\reasoning_intervention_nurec_selected_300.jsonl',
    [string]$GroundTruth = 'D:\300_clip_nurec\00_raw\ground_truth\ego_future_gt_nurec_300.jsonl',
    [string]$RuleGroups = 'D:\300_clip_nurec\04_analysis\sensitivity\sensitivity_rule_groups_fixed_alpha05_4800.jsonl',
    [string]$OutputDir = 'D:\300_clip_nurec\04_analysis\ade',
    [double]$NoChangeTolerance = 0.001
)

$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
$culture = [Globalization.CultureInfo]::InvariantCulture

function Key([string]$clip, [string]$mode, [double]$alpha) {
    return "{0}|{1}|{2}" -f $clip, $mode, $alpha.ToString('0.###', $culture)
}

function Read-JsonlMap([string]$Path, [scriptblock]$KeyFn) {
    $map = @{}
    $reader = [System.IO.StreamReader]::new($Path)
    try {
        while (($line = $reader.ReadLine()) -ne $null) {
            if (-not [string]::IsNullOrWhiteSpace($line)) {
                $row = $line | ConvertFrom-Json
                $map[(& $KeyFn $row)] = $row
            }
        }
    } finally { $reader.Dispose() }
    return $map
}

function Get-Ade($prediction, $groundTruth) {
    if ($prediction.Count -ne $groundTruth.Count) { throw 'Waypoint count mismatch' }
    $sum = 0.0
    for ($i = 0; $i -lt $prediction.Count; $i++) {
        $dx = [double]$prediction[$i].x_m - [double]$groundTruth[$i][0]
        $dy = [double]$prediction[$i].y_m - [double]$groundTruth[$i][1]
        $sum += [math]::Sqrt($dx*$dx + $dy*$dy)
    }
    return $sum / $prediction.Count
}

$gt = Read-JsonlMap $GroundTruth { param($r) [string]$r.clip_id }
$groups = Read-JsonlMap $RuleGroups { param($r) Key ([string]$r.clip_id) ([string]$r.mode) ([double]$r.alpha) }
$buckets = @{}
$reader = [System.IO.StreamReader]::new($Intervention)
try {
    while (($line = $reader.ReadLine()) -ne $null) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $r = $line | ConvertFrom-Json
        $clip = [string]$r.clip_id; $mode = [string]$r.mode; $alpha = [double]$r.alpha
        $gkey = Key $clip $mode $alpha
        if (-not $gt.ContainsKey($clip)) { throw "Missing ground truth: $clip" }
        if (-not $groups.ContainsKey($gkey)) { throw "Missing rule group: $gkey" }
        $ground = $gt[$clip].ego_future_xyz
        $baseline = Get-Ade $r.clean_waypoints $ground
        $output = Get-Ade $r.guided_waypoints $ground
        $group = [string]$groups[$gkey].group
        $bucketKey = "$group|$mode|$($alpha.ToString('0.###',$culture))"
        if (-not $buckets.ContainsKey($bucketKey)) { $buckets[$bucketKey] = [System.Collections.Generic.List[object]]::new() }
        $buckets[$bucketKey].Add([pscustomobject]@{group=$group;mode=$mode;alpha=$alpha;baseline=$baseline;output=$output;delta=($output-$baseline)})
    }
} finally { $reader.Dispose() }

$rows = foreach ($bucketKey in ($buckets.Keys | Sort-Object)) {
    $items = @($buckets[$bucketKey]); $n = $items.Count
    $improve = @($items | Where-Object { $_.delta -lt -$NoChangeTolerance }).Count
    $increase = @($items | Where-Object { $_.delta -gt $NoChangeTolerance }).Count
    $same = $n - $improve - $increase
    $sumImprove = ((($items | Where-Object { $_.delta -lt -$NoChangeTolerance } | Measure-Object delta -Sum).Sum) * -1)
    $sumIncrease = (($items | Where-Object { $_.delta -gt $NoChangeTolerance } | Measure-Object delta -Sum).Sum)
    [pscustomobject]@{
        group=$items[0].group; mode=$items[0].mode; alpha=$items[0].alpha; N=$n
        mean_baseline_ADE=[math]::Round((($items|Measure-Object baseline -Average).Average),4)
        mean_output_ADE=[math]::Round((($items|Measure-Object output -Average).Average),4)
        delta_ADE=[math]::Round((($items|Measure-Object delta -Average).Average),4)
        ADE_decrease_n=$improve; ADE_increase_n=$increase; no_change_n=$same
        ADE_decrease_pct=[math]::Round(100*$improve/$n,2)
        ADE_increase_pct=[math]::Round(100*$increase/$n,2)
        no_change_pct=[math]::Round(100*$same/$n,2)
        total_ADE_decrease=[math]::Round($sumImprove,4)
        total_ADE_increase=[math]::Round($sumIncrease,4)
        net_improvement=[math]::Round(($sumImprove-$sumIncrease),4)
    }
}

$csv = Join-Path $OutputDir 'ade_by_rule_group_mode_alpha.csv'
$json = Join-Path $OutputDir 'ade_by_rule_group_mode_alpha.json'
$md = Join-Path $OutputDir 'ade_by_rule_group_mode_alpha.md'
$rows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $csv
$rows | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 -Path $json
$lines = @('| Group | Mode | Alpha | N | Mean baseline ADE | Mean output ADE | ΔADE | ADE giảm | ADE tăng | Không đổi | Tổng giảm | Tổng tăng | Net |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
foreach ($r in $rows) { $lines += ('| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7}% | {8}% | {9}% | {10} | {11} | {12} |' -f $r.group,$r.mode,$r.alpha,$r.N,$r.mean_baseline_ADE,$r.mean_output_ADE,$r.delta_ADE,$r.ADE_decrease_pct,$r.ADE_increase_pct,$r.no_change_pct,$r.total_ADE_decrease,$r.total_ADE_increase,$r.net_improvement) }
$lines | Set-Content -Encoding UTF8 -Path $md
Write-Output "Saved $($rows.Count) group/mode/alpha rows"
Write-Output $csv
Write-Output $md
