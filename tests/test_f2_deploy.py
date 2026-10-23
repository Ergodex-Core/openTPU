"""The deployment CLI must not expose passcodes through either success-output path."""
import contextlib
import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


class DeployOutputTests(unittest.TestCase):
    def test_passcodes_stay_out_of_create_and_revision_output(self):
        for generated in (False, True):
            for existing in (False, True):
                with self.subTest(generated=generated, existing=existing):
                    self.check_output(generated, existing)

    def check_output(self, generated, existing):
        secret = "synthetic-passcode-for-output-regression"
        ec2, core, web, s3 = (mock.Mock() for _ in range(4))
        ec2.describe_instances.return_value = {"Reservations": [{"Instances": [{
            "VpcId": "vpc-test", "SubnetId": "subnet-test",
            "PrivateIpAddress": "192.0.2.1", "SecurityGroups": [{"GroupId": "sg-test"}],
        }]}]}
        core.list_network_connectors.return_value = {"NetworkConnectors": [
            {"Name": "opentpu-f2-vpc", "Arn": "arn:test:connector"}]}
        core.get_network_connector.return_value = {"State": "ACTIVE"}
        web.list_web_functions.return_value = {"functions": [
            {"functionName": "opentpu-f2-web"}] if existing else []}
        # A service response may echo the submitted environment. It is not safe to dump it.
        response = {"revisionConfig": {"serviceConfig": {
            "environmentVariables": {"PASSCODE": secret}}}}
        web.create_web_function.return_value = response
        web.create_web_function_revision.return_value = response
        web.list_web_function_endpoints.return_value = {"endpoints": [
            {"domainName": "demo.example.invalid"}]}
        clients = {"ec2": ec2, "iam": mock.Mock(), "s3": s3,
                   "lambda-core": core, "lambda-web": web}
        session = mock.Mock()
        session.client.side_effect = clients.__getitem__
        boto3 = types.ModuleType("boto3")
        boto3.Session = mock.Mock(return_value=session)
        botocore = types.ModuleType("botocore")
        exceptions = types.ModuleType("botocore.exceptions")
        exceptions.ClientError = type("ClientError", (Exception,), {})
        path = Path(__file__).resolve().parents[1] / "tools/f2_web/deploy.py"
        spec = importlib.util.spec_from_file_location("f2_deploy_output_test", path)
        deploy = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"boto3": boto3, "botocore": botocore,
                                           "botocore.exceptions": exceptions}):
            spec.loader.exec_module(deploy)
        args = ["--instance", "i-test", "--bucket", "test-bucket"]
        if not generated:
            args += ["--passcode", secret]
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(deploy, "sg", return_value="sg-test"), \
                mock.patch.object(deploy, "role", return_value="arn:test:role"), \
                mock.patch.object(deploy, "bundle", return_value=b"test-code"), \
                mock.patch.object(deploy.secrets, "token_urlsafe", return_value=secret), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            deploy.main(args)
        output = stdout.getvalue() + stderr.getvalue()
        self.assertTrue(secret not in output, "deployment output exposed a synthetic passcode")
        self.assertIn("https://demo.example.invalid", stdout.getvalue())
        call = (web.create_web_function_revision if existing else web.create_web_function).call_args
        config = call.kwargs if existing else call.kwargs["revisionConfig"]
        self.assertEqual(config["serviceConfig"]["environmentVariables"]["PASSCODE"], secret)


if __name__ == "__main__":
    unittest.main()
