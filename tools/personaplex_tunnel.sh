#!/usr/bin/env bash
# Run on the workstation after the FPGA service is listening on the instance.
set -euo pipefail
instance_id=${1:-i-0c78de2253f66519d}
local_port=${2:-8998}
remote_port=${3:-8998}
for port in "$local_port" "$remote_port"; do
  if [[ ! "$port" =~ ^[0-9]+$ ]] || (( 10#$port < 1 || 10#$port > 65535 )); then
    echo 'Ports must be integers from 1 to 65535.' >&2
    exit 2
  fi
done
command -v session-manager-plugin >/dev/null || {
  echo 'AWS Session Manager plugin is required on the workstation.' >&2
  exit 1
}
echo "Opening http://localhost:$local_port through SSM to $instance_id:$remote_port"
exec aws ssm start-session --region us-east-1 --target "$instance_id" \
  --document-name AWS-StartPortForwardingSession \
  --parameters "{\"portNumber\":[\"$remote_port\"],\"localPortNumber\":[\"$local_port\"]}"
