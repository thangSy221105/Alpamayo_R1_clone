param(
    [string]$InputCsv = 'D:\300_clip_nurec\04_analysis\ade\ade_by_rule_group_mode_alpha.csv',
    [string]$OutputDir = 'D:\300_clip_nurec\04_analysis\ade'
)

$ErrorActionPreference = 'Stop'
$rows = Import-Csv -LiteralPath $InputCsv
$groups = $rows | Group-Object group,alpha
$out = foreach ($g in $groups) {
    $items = @($g.Group)
    $n = ($items | Measure-Object N -Sum).Sum
    $baselineSum = (($items | ForEach-Object { [double]$_.N * [double]$_.mean_baseline_ADE }) | Measure-Object -Sum).Sum
    $outputSum = (($items | ForEach-Object { [double]$_.N * [double]$_.mean_output_ADE }) | Measure-Object -Sum).Sum
    $dec = ($items | Measure-Object ADE_decrease_n -Sum).Sum
    $inc = ($items | Measure-Object ADE_increase_n -Sum).Sum
    $same = ($items | Measure-Object no_change_n -Sum).Sum
    $totalDec = ($items | Measure-Object total_ADE_decrease -Sum).Sum
    $totalInc = ($items | Measure-Object total_ADE_increase -Sum).Sum
    [pscustomobject]@{
        group=$items[0].group; alpha=[double]$items[0].alpha; N=$n
        mean_baseline_ADE=[math]::Round($baselineSum/$n,4)
        mean_output_ADE=[math]::Round($outputSum/$n,4)
        delta_ADE=[math]::Round(($outputSum-$baselineSum)/$n,4)
        ADE_decrease_n=$dec; ADE_increase_n=$inc; no_change_n=$same
        ADE_decrease_pct=[math]::Round(100*$dec/$n,2)
        ADE_increase_pct=[math]::Round(100*$inc/$n,2)
        no_change_pct=[math]::Round(100*$same/$n,2)
        total_ADE_decrease=[math]::Round($totalDec,4)
        total_ADE_increase=[math]::Round($totalInc,4)
        net_improvement=[math]::Round($totalDec-$totalInc,4)
    }
}
$out = $out | Sort-Object group,alpha
$csv = Join-Path $OutputDir 'ade_by_rule_group_alpha.csv'
$json = Join-Path $OutputDir 'ade_by_rule_group_alpha.json'
$md = Join-Path $OutputDir 'ade_by_rule_group_alpha.md'
$out | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $csv
$out | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 -Path $json
$lines=@('| Group | Alpha | N | Mean baseline ADE | Mean output ADE | ΔADE | ADE giảm | ADE tăng | Không đổi | Tổng giảm | Tổng tăng | Net |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
foreach($r in $out){$lines += ('| {0} | {1} | {2} | {3} | {4} | {5} | {6}% | {7}% | {8}% | {9} | {10} | {11} |' -f $r.group,$r.alpha,$r.N,$r.mean_baseline_ADE,$r.mean_output_ADE,$r.delta_ADE,$r.ADE_decrease_pct,$r.ADE_increase_pct,$r.no_change_pct,$r.total_ADE_decrease,$r.total_ADE_increase,$r.net_improvement)}
$lines | Set-Content -Encoding UTF8 -Path $md
Write-Output "Saved $($out.Count) rows"
Write-Output $md
