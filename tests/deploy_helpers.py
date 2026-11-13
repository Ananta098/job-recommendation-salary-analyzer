"""Load deploy/aws/deploy.py as a module and fake just enough of AWS to exercise it offline."""
import importlib.util
import io
import json
import sys
import types

from tests.conftest import ROOT


def load_deploy():
    spec = importlib.util.spec_from_file_location('deploy_under_test', ROOT / 'deploy' / 'aws' / 'deploy.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ClientError(Exception):
    def __init__(self, code, operation='Call'):
        super().__init__(f'{operation}: {code}')
        self.response = {'Error': {'Code': code, 'Message': code}}


class BotoCoreError(Exception):
    pass


class FakeAWS:
    """In-memory S3 / CloudFormation / EC2 / SSM / CloudWatch Logs."""

    def __init__(self, arn='arn:aws:sts::123456789012:assumed-role/voclabs/user1', stack_status=None):
        self.arn = arn
        self.objects = {}
        self.buckets = set()
        self.calls = []
        self.stack = None if stack_status is None else self._stack(stack_status, {})
        self.instance_state = 'running'
        self.public_ip = '3.80.6.254'
        self.server = {'current': None, 'previous': None}
        self.release_ok = True
        self.denied = set()          # operation names that raise AccessDenied
        self.scripts = []
        self.allowed = ['203.0.113.10/32']

    def _stack(self, status, params):
        return {'StackStatus': status, 'Parameters': [{'ParameterKey': k, 'ParameterValue': v} for k, v in params.items()],
                'Outputs': [{'OutputKey': 'InstanceId', 'OutputValue': 'i-0123456789abcdef0'},
                            {'OutputKey': 'AppUrl', 'OutputValue': 'http://3.80.6.254:8501'},
                            {'OutputKey': 'HealthUrl', 'OutputValue': 'http://3.80.6.254:8501/_stcore/health'}]}

    def install(self, monkeypatch):
        boto3 = types.ModuleType('boto3')
        fake = self

        class Session:
            def __init__(self, **kwargs):
                fake.calls.append(('session', kwargs.get('region_name')))

            def get_credentials(self):
                return object()

            def client(self, name, **kwargs):
                return Client(fake, name)

        boto3.Session = Session
        botocore = types.ModuleType('botocore')
        exceptions = types.ModuleType('botocore.exceptions')
        exceptions.ClientError = ClientError
        exceptions.BotoCoreError = BotoCoreError
        monkeypatch.setitem(sys.modules, 'boto3', boto3)
        monkeypatch.setitem(sys.modules, 'botocore', botocore)
        monkeypatch.setitem(sys.modules, 'botocore.exceptions', exceptions)
        return self

    def called(self, name):
        return [c for c in self.calls if c[0] == name]


class Waiter:
    def __init__(self, fake, name):
        self.fake, self.name = fake, name

    def wait(self, **kwargs):
        self.fake.calls.append(('wait', self.name))
        if self.name == 'stack_delete_complete':
            self.fake.stack = None
        if self.name == 'instance_running':
            self.fake.instance_state = 'running'


class Paginator:
    def __init__(self, fake):
        self.fake = fake

    def paginate(self, Bucket, Prefix):
        keys = sorted(k for (b, k) in self.fake.objects if b == Bucket and k.startswith(Prefix))
        return [{'Contents': [{'Key': k} for k in keys]}]


class Client:
    def __init__(self, fake, service):
        self.fake, self.service = fake, service

    def __getattr__(self, operation):
        fake = self.fake

        def call(*args, **kwargs):
            fake.calls.append((operation, kwargs))
            if operation in fake.denied:
                raise ClientError('AccessDenied', operation)
            handler = getattr(type(self), '_' + operation, None)
            return handler(self, **kwargs) if handler else {}
        return call

    # STS / S3
    def _get_caller_identity(self):
        return {'Account': '123456789012', 'Arn': self.fake.arn}

    def _head_bucket(self, Bucket):
        if Bucket not in self.fake.buckets:
            raise ClientError('404')

    def _create_bucket(self, Bucket, **kwargs):
        self.fake.buckets.add(Bucket)

    def _create_multipart_upload(self, Bucket, Key, **kwargs):
        self.fake.objects[(Bucket, Key)] = b''
        return {'UploadId': 'upload-1'}

    def _upload_part(self, Bucket, Key, Body, PartNumber, **kwargs):
        self.fake.objects[(Bucket, Key)] += Body
        return {'ETag': f'etag-{PartNumber}'}

    def _put_object(self, Bucket, Key, Body, **kwargs):
        self.fake.objects[(Bucket, Key)] = Body

    def _get_object(self, Bucket, Key):
        return {'Body': io.BytesIO(self.fake.objects[(Bucket, Key)])}

    def _delete_object(self, Bucket, Key):
        self.fake.objects.pop((Bucket, Key), None)

    def get_paginator(self, name):
        return Paginator(self.fake)

    def get_waiter(self, name):
        return Waiter(self.fake, name)

    # CloudFormation / EC2
    def _describe_stacks(self, StackName):
        if self.fake.stack is None:
            raise ClientError('ValidationError')
        return {'Stacks': [self.fake.stack]}

    def _create_stack(self, Parameters, **kwargs):
        self.fake.stack = self.fake._stack('CREATE_COMPLETE', {p['ParameterKey']: p['ParameterValue'] for p in Parameters})
        self.fake.server['current'] = dict((p['ParameterKey'], p['ParameterValue']) for p in Parameters)['ReleaseVersion']

    def _describe_instances(self, InstanceIds):
        instance = {'InstanceId': InstanceIds[0], 'InstanceType': 't3.large', 'State': {'Name': self.fake.instance_state}}
        if self.fake.instance_state == 'running':
            instance['PublicIpAddress'] = self.fake.public_ip
        return {'Reservations': [{'Instances': [instance]}]}

    # SSM: a tiny model of the on-server release state
    def _describe_instance_information(self, Filters):
        return {'InstanceInformationList': [{'PingStatus': 'Online'}]}

    def _send_command(self, Parameters, **kwargs):
        script = Parameters['commands'][0]
        self.fake.scripts.append(script)
        return {'Command': {'CommandId': f'cmd-{len(self.fake.scripts)}'}}

    def _get_command_invocation(self, CommandId, InstanceId):
        script = self.fake.scripts[int(CommandId.split('-')[1]) - 1]
        server = self.fake.server
        if 'install-host.sh' in script:
            version = script.split('install-host.sh" \'')[1].split("'")[0]
            if not self.fake.release_ok:
                return {'Status': 'Failed', 'StandardOutputContent': '[release] new version failed; restoring', 'StandardErrorContent': ''}
            if server['current'] != version:
                server['previous'], server['current'] = server['current'], version
            return {'Status': 'Success', 'StandardOutputContent': f'[release] now serving {version}', 'StandardErrorContent': ''}
        if 'current_version' in script:
            out = f"{server['current'] or 'none'}\n{server['previous'] or 'none'}\n"
            return {'Status': 'Success', 'StandardOutputContent': out, 'StandardErrorContent': ''}
        return {'Status': 'Success', 'StandardOutputContent': 'log line', 'StandardErrorContent': ''}

    # Security group (presentation access)
    def _describe_stack_resource(self, StackName, LogicalResourceId):
        return {'StackResourceDetail': {'PhysicalResourceId': 'sg-0123'}}

    def _authorize_security_group_ingress(self, GroupId, IpPermissions):
        cidr = IpPermissions[0]['IpRanges'][0]['CidrIp']
        if cidr in self.fake.allowed:
            raise ClientError('InvalidPermission.Duplicate')
        self.fake.allowed.append(cidr)

    def _revoke_security_group_ingress(self, GroupId, IpPermissions):
        cidr = IpPermissions[0]['IpRanges'][0]['CidrIp']
        if cidr not in self.fake.allowed:
            raise ClientError('InvalidPermission.NotFound')
        self.fake.allowed.remove(cidr)

    def _describe_security_groups(self, GroupIds):
        return {'SecurityGroups': [{'IpPermissions': [{'FromPort': 8501, 'IpRanges': [{'CidrIp': c} for c in self.fake.allowed]}]}]}

    # CloudWatch Logs
    def _filter_log_events(self, **kwargs):
        return {'events': [{'timestamp': 0, 'message': json.dumps({'log': 'Streamlit started\n'})}]}
