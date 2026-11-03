"""Package the app and provision an EC2 stack without any Git push.

Default action prepares a local package/plan; --apply performs AWS changes.
AWS authentication is via the normal SDK chain, never keys in this repository.
"""
import argparse,hashlib,ipaddress,json,tarfile,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def package(root,output):
    if not (root/'artifacts/deployment/manifest.json').exists():raise ValueError('Export a selected-model bundle first')
    import sys
    sys.path.insert(0,str(root))
    from recommender.bundle import check_bundle
    check_bundle(root/'artifacts/deployment',deep=True)
    from recommender.translation import check_translation
    check_translation(root/'artifacts/translation',deep=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    allowed=['Dockerfile.deploy','.dockerignore','requirements-app.txt','requirements-app.lock.txt','app.py','.streamlit/config.toml']
    allowed.extend(str(p.relative_to(root)) for p in (root/'docker').glob('*.sh'))
    for directory in ['recommender','m9_skills']:
        allowed.extend(str(p.relative_to(root)) for p in (root/directory).glob('*.py'))
    allowed.extend(str(p.relative_to(root)) for p in (root/'artifacts/deployment').rglob('*') if p.is_file())
    allowed.extend(str(p.relative_to(root)) for p in (root/'artifacts/translation').rglob('*') if p.is_file() and '.cache' not in p.parts)
    with tarfile.open(output,'w:gz',compresslevel=1) as archive:
        for rel in sorted(allowed):archive.add(root/rel,arcname=rel,recursive=False)
    with output.open('rb') as stream:checksum=hashlib.file_digest(stream,'sha256').hexdigest()
    return checksum


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--package-only',action='store_true')
    p.add_argument('--config',type=Path,help='JSON configuration copied from deployment.example.json')
    p.add_argument('--region');p.add_argument('--bucket');p.add_argument('--allowed-cidr')
    p.add_argument('--stack-name');p.add_argument('--instance-type',choices=['t3.medium','t3.large','t3.xlarge'])
    p.add_argument('--apply',action='store_true');p.add_argument('--profile')
    a=p.parse_args()
    if a.config:
        config_path=a.config.expanduser()
        if not config_path.is_absolute():config_path=ROOT/config_path
        config=json.loads(config_path.read_text())
        for key in ['region','bucket','allowed_cidr','stack_name','instance_type','profile']:
            if getattr(a,key) is None:setattr(a,key,config.get(key) or None)
    a.stack_name=a.stack_name or 'job-recommender';a.instance_type=a.instance_type or 't3.large'
    if a.instance_type not in ['t3.medium','t3.large','t3.xlarge']:p.error('Unsupported instance type')
    root=a.root.expanduser()
    if not root.is_absolute():root=ROOT/root
    root=root.resolve();out=root/'artifacts/ec2-deployment.tar.gz'
    if a.package_only:
        checksum=package(root,out)
        print(json.dumps({'package':str(out),'sha256':checksum}));return
    if not (a.region and a.bucket and a.allowed_cidr):p.error('--region, --bucket and --allowed-cidr are required for an EC2 plan')
    cidr=str(ipaddress.ip_network(a.allowed_cidr,strict=False))
    if ':' in cidr:p.error('Use an IPv4 CIDR for this template')
    checksum=package(root,out)
    key=f'{a.stack_name}/{checksum}.tar.gz'
    params={'ArtifactBucket':a.bucket,'ArtifactKey':key,'ArtifactSha256':checksum,'ClientCidr':cidr,'InstanceType':a.instance_type}
    plan={'stack':a.stack_name,'region':a.region,'parameters':params,'package':str(out),'apply':a.apply}
    (root/'artifacts/ec2-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(plan,indent=2))
    if not a.apply:return
    import boto3
    from botocore.exceptions import ClientError,NoCredentialsError
    session=boto3.Session(profile_name=a.profile,region_name=a.region)
    if session.get_credentials() is None:raise NoCredentialsError()
    session.client('sts').get_caller_identity() # Authenticate before uploading or changing infrastructure.
    s3=session.client('s3');s3.head_bucket(Bucket=a.bucket)
    s3.upload_file(str(out),a.bucket,key,ExtraArgs={'ServerSideEncryption':'AES256'})
    cfn=session.client('cloudformation')
    request={'StackName':a.stack_name,'TemplateBody':(Path(__file__).parent/'ec2-stack.json').read_text(),
             'Parameters':[{'ParameterKey':k,'ParameterValue':v} for k,v in params.items()],
             'Capabilities':['CAPABILITY_IAM'],'Tags':[{'Key':'Project','Value':'JobRecommendation'}]}
    try:
        cfn.describe_stacks(StackName=a.stack_name);exists=True
    except ClientError as error:
        if error.response['Error']['Code']!='ValidationError':raise
        exists=False
    if exists:raise ValueError('Stack already exists. Use a new stack name; user-data updates do not redeploy an existing application reliably.')
    cfn.create_stack(**request,OnFailure='ROLLBACK')
    print('Stack submitted. Waiting for infrastructure; application bootstrap continues afterward.',flush=True)
    cfn.get_waiter('stack_create_complete').wait(StackName=a.stack_name,WaiterConfig={'Delay':15,'MaxAttempts':120})
    stack=cfn.describe_stacks(StackName=a.stack_name)['Stacks'][0]
    outputs={v['OutputKey']:v['OutputValue'] for v in stack.get('Outputs',[])}
    (root/'artifacts/ec2-status.json').write_text(json.dumps(outputs,indent=2)+'\n')
    print(json.dumps(outputs,indent=2))
    print('CloudFormation completion is not application-health confirmation. Check the health URL after bootstrap; use SSM for logs.')

if __name__=='__main__':main()
