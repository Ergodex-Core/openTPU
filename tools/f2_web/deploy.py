#!/usr/bin/env python3
"""Deploy the F2 demo page (tools/f2_web/lambda) as a Lambda web function in the F2 host's VPC.

    python tools/f2_web/deploy.py --instance i-0c78de2253f66519d --bucket BUCKET [--passcode P]

Idempotent: what exists (by name) is reused. It creates or updates
  - security groups: opentpu-f2-webfn (the function's network interfaces) and opentpu-f2-gw
    (port 8080 from the first only), the second added to the instance beside its own;
  - IAM roles: opentpu-f2-web-exec (the function: CloudWatch logs) and
    opentpu-f2-netconn (the network connector: the function's network interfaces in the VPC);
  - a Lambda network connector (VPC egress into the instance's subnet);
  - the code (index.js, package.json, public/) zipped to s3://BUCKET/opentpu-f2/web/<sha>.zip;
  - the web function opentpu-f2-web, a revision with GATEWAY_URL (the instance's private
    address, port 8080) and PASSCODE, and its ApplicationManaged endpoint (two environments, the minimum, a
    rate limit): the page's URL is printed.
The gateway (tools/f2_web/gateway.py) runs on the instance. Credentials: the environment's.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import secrets
import time
import zipfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

HERE = Path(__file__).resolve().parent
NAME = "opentpu-f2-web"
LAMBDA_TRUST = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Action": "sts:AssumeRole",
                "Principal": {"Service": "lambda.amazonaws.com"}}]}
ENI_POLICY = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Resource": "*", "Action": [
    "ec2:CreateNetworkInterface", "ec2:DeleteNetworkInterface", "ec2:DescribeNetworkInterfaces",
    "ec2:DescribeSubnets", "ec2:DescribeSecurityGroups", "ec2:DescribeVpcs",
    "ec2:AssignPrivateIpAddresses", "ec2:UnassignPrivateIpAddresses",
    "ec2:CreateNetworkInterfacePermission", "ec2:DeleteNetworkInterfacePermission"]}]}


def bundle() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((HERE / "lambda").rglob("*")):
            if p.is_file():
                info = zipfile.ZipInfo(str(p.relative_to(HERE / "lambda")), (2026, 1, 1, 0, 0, 0))
                info.external_attr = 0o644 << 16
                z.writestr(info, p.read_bytes())
    return buf.getvalue()


def sg(ec2, vpc: str, name: str, desc: str) -> str:
    r = ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": [name]},
                                              {"Name": "vpc-id", "Values": [vpc]}])["SecurityGroups"]
    if r:
        return r[0]["GroupId"]
    gid = ec2.create_security_group(GroupName=name, Description=desc, VpcId=vpc,
        TagSpecifications=[{"ResourceType": "security-group",
                            "Tags": [{"Key": "Project", "Value": "openTPU"}]}])["GroupId"]
    return gid


def role(iam, name: str, managed=(), inline: dict | None = None) -> str:
    try:
        arn = iam.get_role(RoleName=name)["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        arn = iam.create_role(RoleName=name, AssumeRolePolicyDocument=json.dumps(LAMBDA_TRUST),
                              Tags=[{"Key": "Project", "Value": "openTPU"}])["Role"]["Arn"]
        time.sleep(10)                                  # IAM propagation
    for m in managed:
        iam.attach_role_policy(RoleName=name, PolicyArn=m)
    if inline:
        iam.put_role_policy(RoleName=name, PolicyName=name, PolicyDocument=json.dumps(inline))
    return arn


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--instance", required=True)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--passcode", help="default: a new random one (printed)")
    ap.add_argument("--rate-limit", type=int, default=5, help="endpoint requests per second")
    a = ap.parse_args(argv)
    s = boto3.Session(region_name=a.region)
    ec2, iam, s3 = s.client("ec2"), s.client("iam"), s.client("s3")
    core, web = s.client("lambda-core"), s.client("lambda-web")

    inst = ec2.describe_instances(InstanceIds=[a.instance])["Reservations"][0]["Instances"][0]
    vpc, subnet, ip = inst["VpcId"], inst["SubnetId"], inst["PrivateIpAddress"]
    fn_sg = sg(ec2, vpc, "opentpu-f2-webfn", "openTPU F2 demo: web function network interfaces")
    gw_sg = sg(ec2, vpc, "opentpu-f2-gw", "openTPU F2 demo: the gateway, from the web function")
    try:
        ec2.authorize_security_group_ingress(GroupId=gw_sg, IpPermissions=[{"IpProtocol": "tcp",
            "FromPort": 8080, "ToPort": 8080, "UserIdGroupPairs": [{"GroupId": fn_sg}]}])
    except ClientError as e:
        if e.response["Error"]["Code"] != "InvalidPermission.Duplicate":
            raise
    groups = [g["GroupId"] for g in inst["SecurityGroups"]]
    if gw_sg not in groups:
        ec2.modify_instance_attribute(InstanceId=a.instance, Groups=groups + [gw_sg])
    print(f"security groups: function {fn_sg}, gateway {gw_sg} (instance {a.instance}, {ip})")

    exec_arn = role(iam, "opentpu-f2-web-exec",
                    ["arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"])
    op_arn = role(iam, "opentpu-f2-netconn", inline=ENI_POLICY)
    print(f"roles: {exec_arn}, {op_arn}")

    conns = [c for c in core.list_network_connectors().get("NetworkConnectors", [])
             if c.get("Name") == "opentpu-f2-vpc"]
    if conns:
        conn_arn = conns[0]["Arn"]
    else:
        conn_arn = core.create_network_connector(Name="opentpu-f2-vpc", OperatorRole=op_arn,
            Configuration={"VpcEgressConfiguration": {"SubnetIds": [subnet],
                                                      "SecurityGroupIds": [fn_sg],
                                                      "NetworkProtocol": "IPv4",
                                                      "AssociatedComputeResourceTypes": ["WebFunction"]}},
            Tags={"Project": "openTPU"})["Arn"]
    for _ in range(90):
        st = core.get_network_connector(Identifier=conn_arn)
        state = st.get("State", "")
        if state.upper() in ("ACTIVE", "AVAILABLE", "READY"):
            break
        if state.upper() in ("FAILED", "INACTIVE"):
            raise RuntimeError(f"network connector {state}: {st.get('StateReason')}")
        time.sleep(10)
    print(f"network connector: {conn_arn} ({state})")

    code = bundle()
    key = f"opentpu-f2/web/{hashlib.sha256(code).hexdigest()[:16]}.zip"
    s3.put_object(Bucket=a.bucket, Key=key, Body=code)
    passcode = a.passcode or secrets.token_urlsafe(12)
    rev = {"buildConfig": {"codeConfig": {"s3Object": {"bucket": a.bucket, "key": key}},
                           "runtimeConfig": {"runtime": "nodejs24.x"}},
           "serviceConfig": {"executionRoleArn": exec_arn, "timeoutSeconds": 900,
                             "egressNetworkConnectorArn": conn_arn,
                             "maxConcurrencyPerEnvironment": 16,
                             "environmentVariables": {"GATEWAY_URL": f"http://{ip}:8080",
                                                      "PASSCODE": passcode}}}
    names = [f["functionName"] for f in web.list_web_functions().get("functions", [])]
    if NAME not in names:
        r = web.create_web_function(functionName=NAME, revisionConfig=rev,
            endpointConfig={"endpointName": "dev", "endpointType": "HomeRegion",
                            "authType": "ApplicationManaged", "autoDeploymentMode": "LatestRevision",
                            "scalingConfig": {"maxEnvironments": 2},
                            "throttleConfig": {"rateLimit": a.rate_limit}},
            tags={"Project": "openTPU"})
    else:
        r = web.create_web_function_revision(functionName=NAME, **rev)
    print(json.dumps(r, indent=1, default=str)[:3000])
    for _ in range(60):
        eps = web.list_web_function_endpoints(functionName=NAME).get("endpoints", [])
        url = next((f"https://{e['domainName']}" for e in eps if e.get("domainName")), None)
        if url:
            break
        time.sleep(5)
    print(f"\npage: {url}\npasscode: {passcode}")


if __name__ == "__main__":
    main()
