"""deploy.py command flows against an in-memory AWS. Nothing here contacts AWS."""
import json

import pytest

from tests.deploy_helpers import FakeAWS, load_deploy

ENV = '''AWS_ACCESS_KEY_ID=ASIAFAKEFAKEFAKE
AWS_SECRET_ACCESS_KEY=fake-secret-value-never-printed
AWS_SESSION_TOKEN=fake-session-token-never-printed
AWS_DEFAULT_REGION=us-east-1
STACK_NAME=job-recommender
ALLOWED_CIDR=203.0.113.10/32
'''


@pytest.fixture()
def project(tmp_path, monkeypatch):
    root = tmp_path / 'project'
    (root / 'artifacts').mkdir(parents=True)
    (root / 'app.py').write_text('print("app")\n')
    (root / '.env.aws').write_text(ENV)
    deploy = load_deploy()
    monkeypatch.setattr(deploy, 'allowlist', lambda r: ['app.py'])
    monkeypatch.setattr(deploy, 'wait_for_app', lambda *a, **k: True)
    monkeypatch.setattr(deploy, 'check_health', lambda url: True)
    monkeypatch.setattr(deploy, 'git_info', lambda r: 'abc1234')
    monkeypatch.setattr(deploy.time, 'sleep', lambda s: None)
    for key in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN', 'AWS_DEFAULT_REGION', 'AWS_PROFILE']:
        monkeypatch.delenv(key, raising=False)
    return deploy, root


def run(deploy, root, *args):
    return deploy.main(['--root', str(root), *args])


def status(root):
    return json.loads((root / 'artifacts' / 'ec2-status.json').read_text())


def test_placeholders_block_every_aws_command(project):
    deploy, root = project
    (root / '.env.aws').write_text('AWS_ACCESS_KEY_ID=PASTE_YOUR_ACCESS_KEY_ID\nAWS_SECRET_ACCESS_KEY=PASTE_X\n')
    for flag in ['--apply', '--update', '--rollback', '--status', '--versions', '--logs', '--destroy']:
        with pytest.raises(SystemExit, match='placeholder'):
            run(deploy, root, flag)


def test_plan_changes_nothing_and_shows_cost(project, capsys, monkeypatch):
    deploy, root = project
    fake = FakeAWS().install(monkeypatch)
    run(deploy, root)
    out = capsys.readouterr().out
    assert fake.calls == []
    plan = json.loads((root / 'artifacts' / 'ec2-plan.json').read_text())
    assert plan['estimated_cost_usd']['per_hour'] > 0.08 and plan['git_commit'] == 'abc1234'
    assert 'fake-secret' not in out and 'ASIAFAKE' not in out and 'fake-session' not in out


def test_first_deploy_versions_package_and_uses_lab_profile(project, capsys, monkeypatch):
    deploy, root = project
    fake = FakeAWS().install(monkeypatch)
    run(deploy, root, '--apply')
    params = {p['ParameterKey']: p['ParameterValue'] for p in fake.called('create_stack')[0][1]['Parameters']}
    assert params['ExistingInstanceProfile'] == 'LabInstanceProfile'
    version = params['ReleaseVersion']
    assert version.endswith('-abc1234') and params['ArtifactKey'] == f'job-recommender/packages/{version}.tar.gz'
    bucket = params['ArtifactBucket']
    meta = json.loads(fake.objects[(bucket, f'job-recommender/versions/{version}.json')])
    assert meta['sha256'] == params['ArtifactSha256'] and meta['commit'] == 'abc1234'
    saved = status(root)
    assert saved['Version'] == version and saved['History'][-1]['action'] == 'deploy'
    assert saved['AppUrl'] == 'http://3.80.6.254:8501'
    assert fake.called('create_log_group') and not fake.called('put_metric_alarm')
    out = capsys.readouterr().out
    assert 'fake-secret' not in out and 'fake-session' not in out and 'ASIAFAKE' not in json.dumps(saved)


def test_failed_rollback_complete_stack_is_cleared_then_recreated(project, capsys, monkeypatch):
    deploy, root = project
    fake = FakeAWS(stack_status='ROLLBACK_COMPLETE').install(monkeypatch)
    run(deploy, root, '--apply')
    assert fake.called('delete_stack') and fake.called('create_stack')
    assert 'ROLLBACK_COMPLETE' in capsys.readouterr().out


@pytest.mark.parametrize('state,message', [('CREATE_COMPLETE', 'aws-update'), ('CREATE_IN_PROGRESS', 'busy'),
                                           ('DELETE_FAILED', 'aws-destroy')])
def test_existing_stack_gives_clear_next_step(project, monkeypatch, state, message):
    deploy, root = project
    FakeAWS(stack_status=state).install(monkeypatch)
    with pytest.raises(SystemExit, match=message):
        run(deploy, root, '--apply')


def deployed(project, monkeypatch, **kwargs):
    deploy, root = project
    fake = FakeAWS(**kwargs).install(monkeypatch)
    run(deploy, root, '--apply')
    return deploy, root, fake


def test_update_installs_new_version_in_place(project, monkeypatch):
    deploy, root, fake = deployed(project, monkeypatch)
    first = status(root)['Version']
    monkeypatch.setattr(deploy, 'new_version', lambda commit, now=None: '20990101-000000-abc1234')
    run(deploy, root, '--update')
    assert not [c for c in fake.called('create_stack')][1:], 'update must not create a second stack'
    assert 'install-host.sh" \'20990101-000000-abc1234\'' in fake.scripts[-1]
    assert 'sha256sum --check' in fake.scripts[-1]
    saved = status(root)
    assert saved['Version'] == '20990101-000000-abc1234' and [h['action'] for h in saved['History']] == ['deploy', 'update']
    assert fake.server == {'current': '20990101-000000-abc1234', 'previous': first}


def test_failed_update_keeps_recorded_version(project, monkeypatch):
    deploy, root, fake = deployed(project, monkeypatch)
    first = status(root)['Version']
    fake.release_ok = False
    monkeypatch.setattr(deploy, 'new_version', lambda commit, now=None: '20990101-000000-abc1234')
    with pytest.raises(SystemExit, match='kept \\(or restored\\) its previous version'):
        run(deploy, root, '--update')
    assert status(root)['Version'] == first


def test_rollback_returns_to_previous_and_checks_health(project, monkeypatch, capsys):
    deploy, root, fake = deployed(project, monkeypatch)
    first = status(root)['Version']
    monkeypatch.setattr(deploy, 'new_version', lambda commit, now=None: '20990101-000000-abc1234')
    run(deploy, root, '--update')
    run(deploy, root, '--rollback')
    assert fake.server['current'] == first
    assert status(root)['Version'] == first and status(root)['History'][-1]['action'] == 'rollback'
    assert 'health check passed' in capsys.readouterr().out


def test_rollback_to_named_and_unknown_version(project, monkeypatch):
    deploy, root, fake = deployed(project, monkeypatch)
    first = status(root)['Version']
    with pytest.raises(SystemExit, match='not stored'):
        run(deploy, root, '--rollback', '20000101-000000-nothere')
    run(deploy, root, '--rollback', first)
    assert fake.server['current'] == first


def test_legacy_stack_package_is_registered_for_rollback(project, monkeypatch):
    deploy, root = project
    fake = FakeAWS().install(monkeypatch)
    fake.stack = fake._stack('CREATE_COMPLETE', {'ArtifactBucket': 'job-recommender-123456789012-us-east-1',
                                                 'ArtifactKey': 'job-recommender/20261111-152621.tar.gz',
                                                 'ArtifactSha256': 'a' * 64})
    fake.buckets.add('job-recommender-123456789012-us-east-1')
    run(deploy, root, '--versions')
    assert ('job-recommender-123456789012-us-east-1', 'job-recommender/versions/20261111-152621.json') in fake.objects


def test_stopped_learner_lab_server_is_started_before_update(project, monkeypatch):
    deploy, root, fake = deployed(project, monkeypatch)
    fake.instance_state = 'stopped'
    run(deploy, root, '--update')
    assert fake.called('start_instances')


def test_log_groups_degrade_without_blocking_deploy(project, monkeypatch, capsys):
    deploy, root = project
    fake = FakeAWS().install(monkeypatch)
    fake.denied = {'create_log_group'}
    run(deploy, root, '--apply')
    out = capsys.readouterr().out
    assert 'Deployed version' in out and 'CloudWatch logs: skipped (AccessDenied)' in out


def test_status_logs_versions_and_destroy(project, monkeypatch, capsys):
    deploy, root, fake = deployed(project, monkeypatch)
    run(deploy, root, '--status')
    run(deploy, root, '--logs')
    run(deploy, root, '--versions')
    out = capsys.readouterr().out
    assert 'Health: OK' in out and 'Streamlit started' in out and '* ' in out
    run(deploy, root, '--destroy')
    assert fake.stack is None and not [k for (_, k) in fake.objects if k.startswith('job-recommender/')]
    assert not (root / 'artifacts' / 'ec2-status.json').exists()


def test_shell_values_are_validated(project):
    deploy, _ = project
    for kind, bad in [('version', "1; rm -rf /"), ('key', "a b"), ('sha', 'xyz'), ('bucket', 'Bad_Bucket')]:
        with pytest.raises(SystemExit, match='Unsafe'):
            deploy.safe(kind, bad)
    assert deploy.new_version('abc1234-dirty', now=0) == '19700101-000000-abc1234-dirty'


def test_cancelled_learner_lab_keys_give_clear_message(project, monkeypatch):
    from tests.deploy_helpers import ClientError
    deploy, root = project
    fake = FakeAWS(stack_status='CREATE_COMPLETE').install(monkeypatch)

    def denied(*args, **kwargs):
        error = ClientError('AccessDenied', 'DescribeStacks')
        error.args = ('User is not authorized ... with an explicit deny in an identity-based policy: '
                      'arn:aws:iam::123456789012:policy/voc-cancel-cred',)
        raise error
    monkeypatch.setattr(deploy.Deployment, 'stack_info', denied)
    with pytest.raises(SystemExit, match='Learner Lab keys were cancelled'):
        deploy.run_cli(['--root', str(root), '--apply'])
