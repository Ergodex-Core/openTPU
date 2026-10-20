#!/bin/bash
# usage: path_analysis.sh <stage dir with route_timing.rpt + worst_paths.rpt>
# (report_timing_summary boxes its section titles as "| Intra Clock Table" and indents the rows)
D="$1"; T="$D/route_timing.rpt"; W="$D/worst_paths.rpt"
echo "== $D  ($(date -u +%H:%M))"
echo "-- design timing summary (WNS TNS failing total WHS THS ...)"; sed -n '/^| Design Timing Summary/,/^| Clock Summary/p' "$T" | grep -E "^ +-?[0-9]+\.[0-9]+ " | head -2
echo "-- intra-clock table (clock WNS TNS failing total WHS THS)"; sed -n '/^| Intra Clock Table/,/^| Inter Clock Table/p' "$T" | awk 'NF>=7 && $2 ~ /^-?[0-9]+\.[0-9]+$/ {printf "%-40s WNS=%8s TNS=%12s fail=%7s total=%8s WHS=%8s THS=%10s\n", $1, $2, $3, $4, $5, $6, $7}' | head -12
echo "-- inter-clock table, violated only (from to WNS TNS failing)"; sed -n '/^| Inter Clock Table/,/^| Other Path Groups Table/p' "$T" | awk 'NF>=8 && $3 ~ /^-?[0-9]+\.[0-9]+$/ && $3+0 < 0 {printf "%-28s -> %-28s WNS=%8s TNS=%12s fail=%7s\n", $1, $2, $3, $4, $5}' | head -12
echo "-- worst paths: slack / source / destination (first 50)"
awk '/^Slack \(VIOLATED\)|^Slack \(MET\)/ {slack=$0; sub(/.*: */,"",slack); sub(/ .*/,"",slack)} /^ +Source: / {src=$2} /^ +Destination: / {dst=$2} /^ +Requirement: / {req=$2} /^ +Data Path Delay: / {dpd=$4} /^ +Logic Levels: / {ll=$3; printf "%s req=%s dpd=%s LL=%s\n   %s\n   -> %s\n", slack, req, dpd, ll, src, dst}' "$W" | head -90
echo "-- worst-50 source/destination block histogram (first 4 hierarchy levels)"
awk '/^ +Source: / {s=$2; n=split(s,a,"/"); k=a[1]"/"a[2]"/"a[3]"/"a[4]; src[k]++} /^ +Destination: / {s=$2; n=split(s,a,"/"); k=a[1]"/"a[2]"/"a[3]"/"a[4]; dst[k]++} END {print "sources:"; for (k in src) printf "  %3d %s\n", src[k], k; print "destinations:"; for (k in dst) printf "  %3d %s\n", dst[k], k}' "$W" | sort -k1,1 -rn | head -24
echo "-- congestion (route_congestion.rpt)"; grep -E -A12 "^Congestion Report|Router Congestion|Max congestion|congestion level" "$D/route_congestion.rpt" 2>/dev/null | grep -E "Level|North|South|East|West|^\| " | head -14 | cut -c1-140
