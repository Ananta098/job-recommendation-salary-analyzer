"""Deploy the app to one EC2 instance, update it in place, roll back, and watch it.

Fill in `.env.aws` (copy of `.env.aws.example`), then use the make targets:

    make aws-plan        dry run: verifies the bundle, prints the plan and cost estimate, changes nothing
    make aws-deploy      first deployment (creates the CloudFormation stack)
    make aws-update      new version on the running server; switched only if it is healthy
    make aws-rollback    back to the previous version (VERSION=<name> for a specific one)
    make aws-versions    versions stored in S3, current and previous marked
    make aws-status      server state, version and health
    make aws-logs        recent application log lines
    make aws-allow CIDR=203.0.113.7/32   let another network open the app (CIDR=me for your current IP)
    make aws-revoke CIDR=203.0.113.7/32  remove that access again
    make aws-destroy     delete the stack and uploaded files

Credentials come only from `.env.aws` or the normal AWS credential chain and are never printed.
"""
import argparse,hashlib,ipaddress,json,os,re,shutil,subprocess,tarfile,time,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
INSTANCE_TYPES=['t3.medium','t3.large','t3.xlarge']
# On-demand Linux prices in us-east-1 / us-west-2 (USD per hour). Check https://aws.amazon.com/ec2/pricing/on-demand/
HOURLY_USD={'t3.medium':0.0416,'t3.large':0.0832,'t3.xlarge':0.1664}
EBS_GB,EBS_USD_PER_GB_MONTH=40,0.08
SECRET_KEYS=('AWS_ACCESS_KEY_ID','AWS_SECRET_ACCESS_KEY','AWS_SESSION_TOKEN')
HOST_FILES=['deploy/aws/release.sh','deploy/aws/install-host.sh','deploy/aws/health-check.sh',
            'deploy/aws/job-recommender.service','deploy/aws/job-recommender-health.service',
            'deploy/aws/job-recommender-health.timer','deploy/aws/cloudwatch-agent.json']
SAFE={'version':r'[A-Za-z0-9][A-Za-z0-9._-]{0,100}','commit':r'[A-Za-z0-9-]{1,48}','bucket':r'[a-z0-9][a-z0-9.-]{1,62}',
      'key':r'[A-Za-z0-9][A-Za-z0-9/._-]{1,300}','sha':r'[0-9a-f]{64}','region':r'[a-z]{2}(-[a-z]+)+-\d','stack':r'[A-Za-z][A-Za-z0-9-]{0,127}'}


class DeployError(SystemExit):
    """A clear, user-facing stop (exit code 1)."""
    def __init__(self,message):super().__init__(message)


def safe(kind,value):
    """Every value placed in a remote shell command must match a strict pattern."""
    if not re.fullmatch(SAFE[kind],str(value)):raise DeployError(f'Unsafe {kind} value refused: {value!r}')
    return str(value)


def load_env(path):
    """Read KEY=VALUE lines; empty values are skipped so they never mask real settings."""
    values={}
    if path.is_file():
        for line in path.read_text(encoding='utf-8').splitlines():
            line=line.strip()
            if not line or line.startswith('#') or '=' not in line:continue
            key,value=line.split('=',1);value=value.strip().strip('"').strip("'")
            if value:values[key.strip()]=value
    return values


def my_cidr():
    try:
        with urllib.request.urlopen('https://checkip.amazonaws.com',timeout=10) as reply:
            address=reply.read().decode().strip()
        return str(ipaddress.ip_address(address))+'/32'
    except (OSError,ValueError) as error:
        raise DeployError('Could not detect your public IP ('+str(error)+'). Set ALLOWED_CIDR in .env.aws, for example 203.0.113.10/32.')


def git_info(root):
    """Short commit of the source tree, with -dirty when there are uncommitted changes."""
    try:
        commit=subprocess.run(['git','rev-parse','--short=7','HEAD'],cwd=root,capture_output=True,text=True,check=True).stdout.strip()
        dirty=subprocess.run(['git','status','--porcelain','--untracked-files=no'],cwd=root,capture_output=True,text=True,check=True).stdout.strip()
        return commit+('-dirty' if dirty else '')
    except (OSError,subprocess.CalledProcessError):
        return 'nogit'


def new_version(commit,now=None):
    return safe('version',time.strftime('%Y%m%d-%H%M%S',time.gmtime(now))+'-'+commit)


def forbidden(rel):
    """Paths that must never leave this computer, whatever the allowlist says."""
    parts=Path(rel).parts
    return (any(p.startswith('.env') or p in ('.hf_token','.aws','.git','.venv') or p.startswith('.venv') for p in parts)
            or parts[0] in ('data','notebooks','results') or rel.startswith('deploy/.state/') or rel.endswith(('.pem','.key','.csv.gz'))
            or (parts[0]=='data' and rel.endswith('.csv')))


def allowlist(root):
    """Verify the bundle and return the only files that may leave this computer."""
    if not (root/'models/recommender/manifest.json').exists():raise DeployError('Model bundle missing: models/recommender (run  make bundle).')
    import sys
    sys.path.insert(0,str(root))
    from recommender.bundle import check_bundle
    check_bundle(root/'models/recommender',deep=True)
    from recommender.translation import check_translation
    check_translation(root/'models/translation',deep=True)
    allowed=['deploy/Dockerfile','.dockerignore','requirements/app.txt','requirements/app.lock.txt','.streamlit/config.toml']+HOST_FILES
    allowed.extend(p.relative_to(root).as_posix() for p in (root/'deploy/docker').glob('*.sh'))
    for directory in ['recommender','recommender/skills','app']:
        allowed.extend(p.relative_to(root).as_posix() for p in (root/directory).glob('*.py'))
    allowed.extend(p.relative_to(root).as_posix() for p in (root/'models/recommender').rglob('*') if p.is_file())
    allowed.extend(p.relative_to(root).as_posix() for p in (root/'models/translation').rglob('*') if p.is_file() and '.cache' not in p.parts)
    missing=[rel for rel in allowed if not (root/rel).is_file()]
    if missing:raise DeployError('Files needed for the server are missing: '+', '.join(missing[:5]))
    blocked=[rel for rel in allowed if forbidden(rel)]
    if blocked:raise DeployError('Refusing to package private files: '+', '.join(blocked))
    return sorted(set(allowed))


def package(root,output):
    """Local package file (only for --package-only; deploys stream straight to S3 instead)."""
    allowed=allowlist(root);output.parent.mkdir(parents=True,exist_ok=True)
    needed=sum((root/rel).stat().st_size for rel in allowed)
    free=shutil.disk_usage(output.parent).free+(output.stat().st_size if output.exists() else 0)
    if free<needed+200*1024*1024:
        raise DeployError(f'Not enough free disk space to build the package: need about {needed/1e9:.1f} GB, have {free/1e9:.1f} GB.')
    with tarfile.open(output,'w:gz',compresslevel=1) as archive:
        for rel in allowed:archive.add(root/rel,arcname=rel,recursive=False)
    with output.open('rb') as stream:checksum=hashlib.file_digest(stream,'sha256').hexdigest()
    return checksum


class S3Stream:
    """File-like writer: sends the archive to S3 in parts while hashing it. Uses no local disk."""
    def __init__(self,s3,bucket,key,part_size=32*1024*1024):
        self.s3,self.bucket,self.key,self.part_size=s3,bucket,key,part_size
        self.buffer=bytearray();self.parts=[];self.hash=hashlib.sha256();self.sent=0
        self.upload=s3.create_multipart_upload(Bucket=bucket,Key=key,ServerSideEncryption='AES256')['UploadId']
    def write(self,data):
        self.hash.update(data);self.buffer.extend(data)
        while len(self.buffer)>=self.part_size:self._send(self.part_size)
        return len(data)
    def _send(self,size):
        chunk=bytes(self.buffer[:size]);del self.buffer[:size];number=len(self.parts)+1
        reply=self.s3.upload_part(Bucket=self.bucket,Key=self.key,UploadId=self.upload,PartNumber=number,Body=chunk)
        self.parts.append({'PartNumber':number,'ETag':reply['ETag']});self.sent+=len(chunk)
        print(f'  uploaded {self.sent/1e6:,.0f} MB',flush=True)
    def finish(self):
        if self.buffer or not self.parts:self._send(len(self.buffer))
        self.s3.complete_multipart_upload(Bucket=self.bucket,Key=self.key,UploadId=self.upload,MultipartUpload={'Parts':self.parts})
        return self.hash.hexdigest()
    def abort(self):
        self.s3.abort_multipart_upload(Bucket=self.bucket,Key=self.key,UploadId=self.upload)


def stream_package(root,allowed,s3,bucket,key):
    stream=S3Stream(s3,bucket,key)
    try:
        with tarfile.open(fileobj=stream,mode='w|gz',compresslevel=1, encoding='utf-8') as archive:
            for rel in allowed:archive.add(root/rel,arcname=rel,recursive=False)
        return stream.finish()
    except BaseException:
        stream.abort();raise


def ensure_bucket(s3,bucket,region):
    """Use the bucket if it is yours; create it private and encrypted if it does not exist."""
    from botocore.exceptions import ClientError
    try:
        s3.head_bucket(Bucket=bucket);return
    except ClientError as error:
        code=str(error.response['Error']['Code'])
        if code not in ('404','NoSuchBucket','NotFound'):
            raise DeployError(f'Bucket {bucket} exists but this account cannot use it ({code}). Set another S3_BUCKET in .env.aws.')
    extra={} if region=='us-east-1' else {'CreateBucketConfiguration':{'LocationConstraint':region}}
    s3.create_bucket(Bucket=bucket,**extra)
    s3.put_public_access_block(Bucket=bucket,PublicAccessBlockConfiguration={
        'BlockPublicAcls':True,'IgnorePublicAcls':True,'BlockPublicPolicy':True,'RestrictPublicBuckets':True})
    print('Created private S3 bucket '+bucket,flush=True)


def wait_for_app(url,minutes=30,quiet=False):
    """The instance installs Docker and builds the image after the stack is created."""
    if not quiet:print(f'Waiting for the app to answer (usually 8-15 minutes on a new server): {url}',flush=True)
    deadline=time.time()+minutes*60
    while True:
        if check_health(url):return True
        if time.time()>=deadline:return False
        time.sleep(20)
        if not quiet:print('  still starting ...',flush=True)


def check_health(url):
    try:
        with urllib.request.urlopen(url,timeout=8) as reply:return reply.status==200
    except OSError:
        return False


class Deployment:
    """All AWS calls for one stack."""
    def __init__(self,args,env,root):
        self.a,self.env,self.root=args,env,root
        self.status_path=root/'deploy/.state/ec2-status.json'
        self.session=None

    # ---------- connection and lookups ----------
    def connect(self):
        import boto3
        from botocore.exceptions import BotoCoreError,ClientError
        a=self.a
        session=boto3.Session(profile_name=a.profile,region_name=a.region)
        if session.get_credentials() is None:
            raise DeployError('No AWS credentials found. Put AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env.aws.')
        try:identity=session.client('sts').get_caller_identity()
        except (BotoCoreError,ClientError) as error:
            raise DeployError('AWS rejected the credentials in .env.aws: '+type(error).__name__+
                              '\nLearner Lab keys expire when the lab session ends: start the lab again and copy fresh keys (all three lines).')
        self.account=identity['Account']
        print('Signed in to AWS account '+self.account+' in '+a.region,flush=True)
        # AWS Academy Learner Lab sessions cannot create IAM roles; they ship a ready-made LabInstanceProfile.
        self.learner_lab='voclabs' in identity.get('Arn','') or 'LabRole' in identity.get('Arn','')
        if self.learner_lab:
            if not a.instance_profile:a.instance_profile='LabInstanceProfile'
            if a.region not in ('us-east-1','us-west-2'):
                raise DeployError('AWS Academy Learner Lab only allows us-east-1 or us-west-2. Set AWS_DEFAULT_REGION=us-east-1 in .env.aws.')
            print('AWS Academy Learner Lab detected: using LabInstanceProfile',flush=True)
        self.session=session
        self.bucket=safe('bucket',a.bucket or f'job-recommender-{self.account}-{a.region}')
        self.stack=safe('stack',a.stack_name)
        return self

    def client(self,name):
        return self.session.client(name)

    def stack_info(self):
        """(status, parameters, outputs) of the stack, or (None, {}, {}) when it does not exist."""
        from botocore.exceptions import ClientError
        try:
            stack=self.client('cloudformation').describe_stacks(StackName=self.stack)['Stacks'][0]
        except ClientError as error:
            if error.response['Error']['Code']=='ValidationError':return None,{},{}
            raise
        params={p['ParameterKey']:p.get('ParameterValue','') for p in stack.get('Parameters',[])}
        outputs={o['OutputKey']:o['OutputValue'] for o in stack.get('Outputs',[])}
        return stack['StackStatus'],params,outputs

    def instance(self,instance_id):
        reservations=self.client('ec2').describe_instances(InstanceIds=[instance_id])['Reservations']
        return reservations[0]['Instances'][0]

    def urls(self,instance):
        ip=instance.get('PublicIpAddress')
        return (f'http://{ip}:8501',f'http://{ip}:8501/_stcore/health') if ip else (None,None)

    def read_status(self):
        try:return json.loads(self.status_path.read_text(encoding='utf-8'))
        except (OSError,ValueError):return {}

    def write_status(self,status):
        self.status_path.parent.mkdir(exist_ok=True)
        self.status_path.write_text(json.dumps(status,indent=2)+'\n', encoding='utf-8')

    def record(self,status,version,commit,action,instance=None):
        now=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        status.update({'Version':version,'Commit':commit,'DeployedAt':now,'Stack':self.stack,'Region':self.a.region,'Bucket':self.bucket})
        if instance is not None:
            app,health=self.urls(instance)
            if app:status.update({'AppUrl':app,'HealthUrl':health})
        history=status.setdefault('History',[])
        history.append({'version':version,'commit':commit,'time':now,'action':action})
        del history[:-30]
        self.write_status(status)

    # ---------- versions in S3 ----------
    def version_key(self,version):return f'{self.stack}/packages/{version}.tar.gz'
    def meta_key(self,version):return f'{self.stack}/versions/{version}.json'

    def put_version(self,version,commit,key,sha,size=None,note=''):
        meta={'version':version,'commit':commit,'key':key,'sha256':sha,'created':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
              'size_bytes':size,'note':note}
        self.client('s3').put_object(Bucket=self.bucket,Key=self.meta_key(version),Body=json.dumps(meta,indent=2).encode(),
                                     ServerSideEncryption='AES256',ContentType='application/json')
        return meta

    def versions(self):
        """Stored versions, oldest first."""
        s3=self.client('s3');found=[]
        for page in s3.get_paginator('list_objects_v2').paginate(Bucket=self.bucket,Prefix=f'{self.stack}/versions/'):
            for item in page.get('Contents',[]):
                body=s3.get_object(Bucket=self.bucket,Key=item['Key'])['Body'].read()
                try:found.append(json.loads(body))
                except ValueError:continue
        return sorted(found,key=lambda m:(m.get('created',''),m['version']))

    def register_initial(self,params):
        """Stacks created before versioning: record their original package so rollback can reach it."""
        key,sha=params.get('ArtifactKey'),params.get('ArtifactSha256')
        if not key or not sha:return None
        version=params.get('ReleaseVersion') or Path(key).name.replace('.tar.gz','')
        version=safe('version',version if version!='initial' else Path(key).name.replace('.tar.gz',''))
        if any(v['version']==version for v in self.versions()):return version
        self.put_version(version,params.get('GitCommit') or 'unknown',key,sha,note='original stack package')
        return version

    # ---------- remote commands (SSM) ----------
    def ssm_ready(self,instance_id,wait_seconds=0):
        ssm=self.client('ssm');deadline=time.time()+wait_seconds
        while True:
            try:
                info=ssm.describe_instance_information(Filters=[{'Key':'InstanceIds','Values':[instance_id]}])['InstanceInformationList']
                if info and info[0].get('PingStatus')=='Online':return True
            except Exception as error:  # permission or API problems: report, do not crash
                print('  Systems Manager not reachable: '+type(error).__name__,flush=True);return False
            if time.time()>=deadline:return False
            time.sleep(15)

    def ssm_run(self,instance_id,script,label,timeout=3600,show=True):
        """Run a shell script as root on the instance; return (ok, stdout)."""
        ssm=self.client('ssm')
        command=ssm.send_command(InstanceIds=[instance_id],DocumentName='AWS-RunShellScript',Comment=label[:100],
                                 Parameters={'commands':[script],'executionTimeout':[str(timeout)]},TimeoutSeconds=600)['Command']['CommandId']
        if show:print(f'{label} (running on the server; this can take several minutes) ...',flush=True)
        deadline=time.time()+timeout+120
        while time.time()<deadline:
            time.sleep(5)
            try:result=ssm.get_command_invocation(CommandId=command,InstanceId=instance_id)
            except Exception as error:
                if 'InvocationDoesNotExist' in type(error).__name__ or 'InvocationDoesNotExist' in str(error):continue
                raise
            if result['Status'] in ('Pending','InProgress','Delayed'):continue
            out=result.get('StandardOutputContent','');err=result.get('StandardErrorContent','')
            if show:
                if out.strip():print(out.rstrip())
                if result['Status']!='Success' and err.strip():print(err.rstrip()[-3000:])
            return result['Status']=='Success',out
        return False,''

    def release_script(self,meta):
        b,r,s=self.bucket,safe('region',self.a.region),self.stack
        v,c,k,h=safe('version',meta['version']),safe('commit',meta.get('commit') or 'unknown'),safe('key',meta['key']),safe('sha',meta['sha256'])
        return f'''set -euo pipefail
BASE=/opt/job-recommender
mkdir -p "$BASE/releases" "$BASE/bin"
printf 'BUCKET=%s\\nSTACK=%s\\nREGION=%s\\n' '{b}' '{s}' '{r}' > "$BASE/instance.env"
DIR="$BASE/releases/{v}"
if [ ! -f "$DIR/deploy/aws/install-host.sh" ]; then
  TMP=$(mktemp /tmp/package.XXXXXX)
  aws s3 cp 's3://{b}/{k}' "$TMP" --region '{r}' --only-show-errors
  echo '{h}  '"$TMP" | sha256sum --check --status || {{ echo 'Package checksum mismatch: refusing to install'; rm -f "$TMP"; exit 1; }}
  rm -rf "$DIR"; mkdir -p "$DIR"; tar -xzf "$TMP" -C "$DIR"; rm -f "$TMP"
fi
bash "$DIR/deploy/aws/install-host.sh" '{v}' '{c}'
'''

    def server_versions(self,instance_id):
        ok,out=self.ssm_run(instance_id,'cat /opt/job-recommender/current_version 2>/dev/null || echo none; '
                                         'cat /opt/job-recommender/previous_version 2>/dev/null || echo none','Reading the deployed version',300,show=False)
        if not ok:return None,None
        lines=[x.strip() for x in out.splitlines() if x.strip()]+['none','none']
        return (None if lines[0]=='none' else lines[0]),(None if lines[1]=='none' else lines[1])

    def running_instance(self,outputs):
        """The instance id; starts a stopped instance (Learner Lab stops servers when a session ends)."""
        instance_id=outputs.get('InstanceId')
        if not instance_id:raise DeployError('The stack has no InstanceId output. Run  make aws-status.')
        instance=self.instance(instance_id);state=instance['State']['Name']
        if state=='stopped':
            print('The server is stopped (Learner Lab stops servers at the end of a session); starting it ...',flush=True)
            ec2=self.client('ec2');ec2.start_instances(InstanceIds=[instance_id])
            ec2.get_waiter('instance_running').wait(InstanceIds=[instance_id],WaiterConfig={'Delay':10,'MaxAttempts':60})
            instance=self.instance(instance_id)
        elif state!='running':
            raise DeployError(f'The server is {state}. Wait a minute and run  make aws-status.')
        if not self.ssm_ready(instance_id,wait_seconds=300):
            raise DeployError('The server does not answer through AWS Systems Manager, so it cannot be updated in place.\n'
                              'Fallback: make aws-destroy  then  make aws-deploy  (short downtime).')
        return instance_id,instance

    # ---------- log groups (optional: skipped with a message if the account forbids it) ----------
    def monitoring(self,instance_id=None):
        """Create the two CloudWatch log groups with a retention period. Never blocks a deploy."""
        from botocore.exceptions import ClientError
        retention=int(self.env.get('LOG_RETENTION_DAYS','14'))
        try:
            client=self.client('logs')
            for suffix in ['app','bootstrap']:
                group=f'/job-recommender/{self.stack}/{suffix}'
                try:client.create_log_group(logGroupName=group)
                except ClientError as error:
                    if error.response['Error']['Code']!='ResourceAlreadyExistsException':raise
                client.put_retention_policy(logGroupName=group,retentionInDays=retention)
            print(f'CloudWatch logs: ok ({retention}-day retention; read them with  make aws-logs)')
            return True
        except ClientError as error:
            print(f'CloudWatch logs: skipped ({error.response["Error"].get("Code","Error")}); make aws-logs reads the server directly instead')
        except Exception as error:
            print(f'CloudWatch logs: skipped ({type(error).__name__})')
        return False

    # ---------- commands ----------
    def deploy(self,allowed,cidr,commit):
        from botocore.exceptions import ClientError
        cfn=self.client('cloudformation')
        status,_,_=self.stack_info()
        if status=='ROLLBACK_COMPLETE':
            print(f'A previous deploy of {self.stack} failed and left an empty stack (ROLLBACK_COMPLETE). Removing it first ...',flush=True)
            cfn.delete_stack(StackName=self.stack)
            cfn.get_waiter('stack_delete_complete').wait(StackName=self.stack,WaiterConfig={'Delay':15,'MaxAttempts':120})
            status=None
        if status is not None:
            if status.endswith('_IN_PROGRESS'):
                raise DeployError(f'Stack {self.stack} is busy ({status}). Wait a few minutes, then run  make aws-status.')
            if status in ('DELETE_FAILED','ROLLBACK_FAILED','UPDATE_ROLLBACK_FAILED','CREATE_FAILED'):
                raise DeployError(f'Stack {self.stack} is in {status} after an earlier failure. Run  make aws-destroy, then  make aws-deploy.')
            raise DeployError(f'Stack {self.stack} is already deployed ({status}). Use  make aws-update  for a new version, '
                              'or  make aws-destroy  first to start over.')
        s3=self.client('s3');ensure_bucket(s3,self.bucket,self.a.region)
        version=new_version(commit);key=self.version_key(version)
        print(f'Uploading version {version} ({self.size/1e9:.1f} GB, straight to S3; no local copy) ...',flush=True)
        sha=stream_package(self.root,allowed,s3,self.bucket,key)
        meta=self.put_version(version,commit,key,sha,self.size)
        params={'ArtifactBucket':self.bucket,'ArtifactKey':key,'ArtifactSha256':sha,'ClientCidr':cidr,'InstanceType':self.a.instance_type,
                'ExistingInstanceProfile':self.a.instance_profile,'ReleaseVersion':version,'GitCommit':safe('commit',commit)}
        cfn.create_stack(StackName=self.stack,TemplateBody=(Path(__file__).parent/'ec2-stack.json').read_text(encoding='utf-8'),
                         Parameters=[{'ParameterKey':k,'ParameterValue':v} for k,v in params.items()],
                         Capabilities=['CAPABILITY_IAM'],Tags=[{'Key':'Project','Value':'JobRecommendation'}],OnFailure='ROLLBACK')
        print('Stack submitted. Creating the server (about 3 minutes) ...',flush=True)
        try:
            cfn.get_waiter('stack_create_complete').wait(StackName=self.stack,WaiterConfig={'Delay':15,'MaxAttempts':120})
        except Exception:
            try:events=cfn.describe_stack_events(StackName=self.stack)['StackEvents']
            except ClientError:events=[]
            reasons=[e['LogicalResourceId']+': '+e.get('ResourceStatusReason','') for e in events if e['ResourceStatus'].endswith('FAILED')]
            raise DeployError('Stack creation failed:\n  '+'\n  '.join(reasons[:5] or ['see the CloudFormation console'])+
                              '\nThe uploaded package is kept. Running  make aws-deploy  again removes the failed stack and retries.')
        _,_,outputs=self.stack_info();instance=self.instance(outputs['InstanceId'])
        status_doc={'InstanceId':outputs['InstanceId'],'AllowedCidr':cidr,'ArtifactKey':key}
        self.record(status_doc,version,commit,'deploy',instance)
        app,health=self.urls(instance)
        healthy=wait_for_app(health)
        self.monitoring(outputs['InstanceId'])
        if healthy:
            print(f'\nDeployed version {version}. Open {app} and submit a profile to confirm recommendations appear.')
            print(f'It is reachable only from {cidr}. The server costs money while it runs: run  make aws-destroy  when finished.')
        else:
            print(f'\nThe server exists but the app did not answer in time. It may still be building; try {app} in a few minutes.')
            print('Then run  make aws-status  and  make aws-logs.')
        return meta

    def release(self,meta,action):
        """Install `meta` on the running server: build/reuse image, test next to the live app, switch if healthy."""
        status,params,outputs=self.stack_info()
        if status is None:raise DeployError(f'Stack {self.stack} does not exist. Run  make aws-deploy  first.')
        if status.endswith('_IN_PROGRESS') or 'FAILED' in status or status=='ROLLBACK_COMPLETE':
            raise DeployError(f'Stack {self.stack} is in {status}; it cannot be updated. Run  make aws-status.')
        instance_id,_=self.running_instance(outputs)
        ok,_=self.ssm_run(instance_id,self.release_script(meta),f'{action.capitalize()}: installing version {meta["version"]}')
        instance=self.instance(instance_id);app,health=self.urls(instance)
        if not ok:
            raise DeployError(f'{action.capitalize()} failed; the server kept (or restored) its previous version. '
                              'Details are above; also see  make aws-logs.')
        if health and not wait_for_app(health,minutes=5,quiet=True):
            raise DeployError(f'The server reports success but {health} does not answer from this computer. '
                              'Check that your IP still matches ALLOWED_CIDR, then run  make aws-status.')
        status_doc=self.read_status();status_doc.setdefault('InstanceId',instance_id)
        self.record(status_doc,meta['version'],meta.get('commit','unknown'),action,instance)
        print(f'\nNow serving version {meta["version"]} at {app} (health check passed).')
        return instance_id

    def update(self,allowed,commit):
        _,params,_=self.stack_info()
        if params:self.register_initial(params)
        version=new_version(commit);key=self.version_key(version)
        print(f'Uploading version {version} ({self.size/1e9:.1f} GB) ...',flush=True)
        sha=stream_package(self.root,allowed,self.client('s3'),self.bucket,key)
        meta=self.put_version(version,commit,key,sha,self.size)
        instance_id=self.release(meta,'update')
        self.monitoring(instance_id)

    def rollback(self,target):
        status,params,outputs=self.stack_info()
        if status is None:raise DeployError(f'Stack {self.stack} does not exist.')
        if params:self.register_initial(params)
        known={m['version']:m for m in self.versions()}
        if target in (None,'','previous'):
            instance_id,_=self.running_instance(outputs)
            current,previous=self.server_versions(instance_id)
            if not previous:
                history=[h['version'] for h in self.read_status().get('History',[]) if h['version']!=current]
                older=[v for v in known if current is None or v<current]
                previous=(history[-1] if history else (older[-1] if older else None))
            if not previous or previous not in known:
                raise DeployError('No previous version is recorded. Run  make aws-versions  and then  make aws-rollback VERSION=<name>.')
            target=previous
            print(f'Rolling back from {current or "unknown"} to {target}',flush=True)
        elif target not in known:
            raise DeployError(f'Version {target} is not stored. Run  make aws-versions  to see the available names.')
        self.release(known[target],'rollback')

    def list_versions(self):
        _,params,outputs=self.stack_info()
        if params:self.register_initial(params)
        versions=self.versions();current=previous=None
        if outputs.get('InstanceId'):
            try:
                if self.instance(outputs['InstanceId'])['State']['Name']=='running' and self.ssm_ready(outputs['InstanceId']):
                    current,previous=self.server_versions(outputs['InstanceId'])
            except Exception:pass
        current=current or self.read_status().get('Version')
        if not versions:print('No versions stored yet for '+self.stack);return versions
        print(f'{"":2s}{"VERSION":34s}{"COMMIT":14s}{"UPLOADED (UTC)":22s}NOTE')
        for m in versions:
            mark='* ' if m['version']==current else ('< ' if m['version']==previous else '  ')
            print(f'{mark}{m["version"]:34s}{(m.get("commit") or "")[:13]:14s}{m.get("created",""):22s}{m.get("note","")}')
        print('\n* = running now, < = previous (target of  make aws-rollback)')
        return versions

    def status(self):
        status,_,outputs=self.stack_info()
        print(f'Stack {self.stack}: {status or "not deployed"}')
        if not outputs.get('InstanceId'):return
        instance=self.instance(outputs['InstanceId']);state=instance['State']['Name']
        app,health=self.urls(instance)
        print(f'Server {outputs["InstanceId"]} ({instance.get("InstanceType","")}): {state}')
        saved=self.read_status();version=saved.get('Version')
        if state=='running' and self.ssm_ready(outputs['InstanceId']):
            version=self.server_versions(outputs['InstanceId'])[0] or version
        print(f'Version: {version or "unknown"}  (commit {saved.get("Commit","unknown")}, deployed {saved.get("DeployedAt","unknown")})')
        if app:
            print(f'App: {app}\nHealth: {"OK" if check_health(health) else "NOT ANSWERING"} ({health})')
            if saved.get('AppUrl') and saved['AppUrl']!=app:
                print('  (the address changed since the last deploy: the server was stopped and started)')
                saved.update({'AppUrl':app,'HealthUrl':health});self.write_status(saved)
        else:print('App: no public address (server not running)')
        print(f'Estimated cost while running: ${HOURLY_USD.get(instance.get("InstanceType",""),0):.4f}/hour for the server')

    def access(self,cidr,allow):
        """Add or remove one address range on the app port of the running stack (no redeploy)."""
        from botocore.exceptions import ClientError
        status,_,_=self.stack_info()
        if status is None:raise DeployError(f'Stack {self.stack} does not exist. Run  make aws-deploy  first.')
        group=self.client('cloudformation').describe_stack_resource(StackName=self.stack,LogicalResourceId='SecurityGroup')['StackResourceDetail']['PhysicalResourceId']
        ec2=self.client('ec2')
        rule=[{'IpProtocol':'tcp','FromPort':8501,'ToPort':8501,'IpRanges':[{'CidrIp':cidr,**({'Description':'added with make aws-allow'} if allow else {})}]}]
        try:
            if allow:ec2.authorize_security_group_ingress(GroupId=group,IpPermissions=rule)
            else:ec2.revoke_security_group_ingress(GroupId=group,IpPermissions=rule)
            print(('Allowed ' if allow else 'Removed ')+cidr+' on port 8501.')
        except ClientError as error:
            code=error.response['Error']['Code']
            if code=='InvalidPermission.Duplicate':print(cidr+' was already allowed.')
            elif code=='InvalidPermission.NotFound':print(cidr+' was not in the allowed list.')
            else:raise
        ranges=[r['CidrIp'] for p in ec2.describe_security_groups(GroupIds=[group])['SecurityGroups'][0].get('IpPermissions',[])
                if p.get('FromPort')==8501 for r in p.get('IpRanges',[])]
        print('Allowed to open the app now: '+(', '.join(ranges) or 'nobody'))
        if '0.0.0.0/0' in ranges:
            print('WARNING: the app is open to the whole internet over plain HTTP. Remove it after the presentation:  make aws-revoke CIDR=0.0.0.0/0')

    def logs(self,minutes):
        from botocore.exceptions import ClientError
        group=f'/job-recommender/{self.stack}/app'
        try:
            events=self.client('logs').filter_log_events(logGroupName=group,startTime=int((time.time()-minutes*60)*1000),limit=200).get('events',[])
            if events:
                print(f'Last {len(events)} app log lines from CloudWatch ({group}, past {minutes} min):')
                for event in events:
                    try:line=json.loads(event['message']).get('log','').rstrip()
                    except ValueError:line=event['message'].rstrip()
                    print(time.strftime('%H:%M:%S',time.gmtime(event['timestamp']/1000)),line)
                return
            print(f'No CloudWatch log lines in the past {minutes} minutes; asking the server directly ...')
        except ClientError as error:
            print(f'CloudWatch logs unavailable ({error.response["Error"]["Code"]}); asking the server directly ...')
        _,_,outputs=self.stack_info()
        if not outputs.get('InstanceId'):raise DeployError('No server to read logs from.')
        if not self.ssm_ready(outputs['InstanceId']):raise DeployError('The server is not reachable through Systems Manager (is it running? see  make aws-status).')
        self.ssm_run(outputs['InstanceId'],'docker logs --tail 100 job-recommender 2>&1 || true; echo "--- bootstrap log ---"; '
                     'tail -n 30 /var/log/job-recommender-bootstrap.log 2>/dev/null || true','Reading recent logs',300)

    def destroy(self):
        from botocore.exceptions import ClientError
        cfn=self.client('cloudformation');status,_,_=self.stack_info()
        if status:
            cfn.delete_stack(StackName=self.stack)
            print('Deleting stack '+self.stack+' (a few minutes) ...',flush=True)
            cfn.get_waiter('stack_delete_complete').wait(StackName=self.stack,WaiterConfig={'Delay':15,'MaxAttempts':120})
        else:print(f'Stack {self.stack} does not exist; cleaning up its stored files only.')
        s3=self.client('s3');removed=0
        try:
            for page in s3.get_paginator('list_objects_v2').paginate(Bucket=self.bucket,Prefix=self.stack+'/'):
                for item in page.get('Contents',[]):
                    s3.delete_object(Bucket=self.bucket,Key=item['Key']);removed+=1
        except ClientError as error:print('Could not clean the bucket: '+error.response['Error']['Code'])
        self.status_path.unlink(missing_ok=True)
        print(f'Removed: stack and {removed} stored file(s) (packages and version records). '
              f'Kept: the empty bucket and the CloudWatch log groups (they expire after their retention period).')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--package-only',action='store_true')
    p.add_argument('--env-file',type=Path,help='Credentials file (default: .env.aws in the project folder)')
    p.add_argument('--region');p.add_argument('--bucket');p.add_argument('--allowed-cidr')
    p.add_argument('--stack-name');p.add_argument('--instance-type',choices=INSTANCE_TYPES);p.add_argument('--profile')
    action=p.add_mutually_exclusive_group()
    action.add_argument('--apply',action='store_true',help='First deployment: create the stack')
    action.add_argument('--update',action='store_true',help='Install a new version on the running server')
    action.add_argument('--rollback',nargs='?',const='previous',metavar='VERSION',help='Return to the previous (or named) version')
    action.add_argument('--versions',action='store_true',help='List stored versions')
    action.add_argument('--status',action='store_true',help='Server state, version and health')
    action.add_argument('--logs',action='store_true',help='Recent application logs');p.add_argument('--minutes',type=int,default=30)
    action.add_argument('--allow',metavar='CIDR',help='Let this address range open the app ("me" = your current IP)')
    action.add_argument('--revoke',metavar='CIDR',help='Remove that address range again')
    action.add_argument('--destroy',action='store_true',help='Delete the stack and stored files')
    a=p.parse_args(argv)
    root=a.root.expanduser()
    if not root.is_absolute():root=ROOT/root
    root=root.resolve();out=root/'deploy/.state/ec2-deployment.tar.gz'
    env_path=(a.env_file.expanduser() if a.env_file else root/'.env.aws')
    env=load_env(env_path)
    touches_aws=any([a.apply,a.update,a.rollback,a.versions,a.status,a.logs,a.destroy,a.allow,a.revoke])
    unfilled=[k for k,v in env.items() if v.startswith('PASTE_')]
    if unfilled and touches_aws:
        raise DeployError(f'{env_path.name} still has placeholder values: '+', '.join(unfilled)+'. Replace them with your own values.')
    for key in SECRET_KEYS+('AWS_DEFAULT_REGION','AWS_PROFILE'):
        if key in env and not env[key].startswith('PASTE_'):os.environ[key]=env[key]
    a.region=a.region or env.get('AWS_DEFAULT_REGION') or os.environ.get('AWS_DEFAULT_REGION') or os.environ.get('AWS_REGION')
    a.bucket=a.bucket or env.get('S3_BUCKET');a.allowed_cidr=a.allowed_cidr or env.get('ALLOWED_CIDR')
    a.stack_name=a.stack_name or env.get('STACK_NAME') or 'job-recommender'
    a.instance_type=a.instance_type or env.get('INSTANCE_TYPE') or 't3.large'
    a.profile=a.profile or os.environ.get('AWS_PROFILE')
    a.instance_profile=env.get('INSTANCE_PROFILE','')
    if a.instance_type not in INSTANCE_TYPES:p.error('Unsupported instance type: choose '+', '.join(INSTANCE_TYPES))
    if a.package_only:
        checksum=package(root,out)
        print(json.dumps({'package':str(out),'sha256':checksum}));return
    if not a.region:p.error('Set AWS_DEFAULT_REGION in .env.aws (for example us-east-1)')
    safe('region',a.region);safe('stack',a.stack_name)
    if a.bucket:safe('bucket',a.bucket)
    d=Deployment(a,env,root)
    if a.versions:return d.connect().list_versions()
    if a.status:return d.connect().status()
    if a.logs:return d.connect().logs(a.minutes)
    if a.destroy:return d.connect().destroy()
    if a.allow or a.revoke:
        value=a.allow or a.revoke
        try:cidr=my_cidr() if value=='me' else str(ipaddress.ip_network(value,strict=False))
        except ValueError:p.error(f'Not an IPv4 address range: {value} (example: 203.0.113.7/32, or 0.0.0.0/0 for everyone)')
        if ':' in cidr:p.error('Use an IPv4 address range')
        return d.connect().access(cidr,allow=bool(a.allow))
    if a.rollback:return d.connect().rollback(None if a.rollback=='previous' else safe('version',a.rollback))
    commit=git_info(root)
    if a.update:
        allowed=allowlist(root);d.size=sum((root/rel).stat().st_size for rel in allowed)
        return d.connect().update(allowed,commit)
    cidr=str(ipaddress.ip_network(a.allowed_cidr or my_cidr(),strict=False))
    if ':' in cidr:p.error('Use an IPv4 CIDR for this template')
    allowed=allowlist(root);d.size=sum((root/rel).stat().st_size for rel in allowed)
    hourly=HOURLY_USD[a.instance_type];storage=EBS_GB*EBS_USD_PER_GB_MONTH/730
    plan={'stack':a.stack_name,'region':a.region,'bucket':a.bucket or '(job-recommender-<account>-<region>, created at deploy)',
          'version':new_version(commit)+' (final name set at upload time)','git_commit':commit,'allowed_cidr':cidr,
          'instance_type':a.instance_type,'files':len(allowed),'upload_size_gb':round(d.size/1e9,2),
          'estimated_cost_usd':{'per_hour':round(hourly+storage,4),'per_8h_day':round((hourly+storage)*8,2),
                                'note':f'{a.instance_type} ${hourly}/h + {EBS_GB} GB gp3 disk; on-demand us-east-1 prices, excludes data transfer'},
          'log_retention_days':int(env.get('LOG_RETENTION_DAYS','14')),'apply':a.apply}
    (root/'deploy/.state').mkdir(parents=True,exist_ok=True)
    (root/'deploy/.state/ec2-plan.json').write_text(json.dumps(plan,indent=2)+'\n', encoding='utf-8')
    print(json.dumps(plan,indent=2))
    if '0.0.0.0/0' in cidr:print('WARNING: ALLOWED_CIDR opens the app to the whole internet over plain HTTP. Use it only for a short demo and revoke it afterwards.')
    if not a.apply:
        print('Plan only: nothing was changed in AWS. Run  make aws-deploy  (first time) or  make aws-update  (new version).');return
    d.connect().deploy(allowed,cidr,commit)


def run_cli(argv=None):
    """Turn AWS permission errors into one clear sentence instead of a traceback."""
    try:
        return main(argv)
    except Exception as error:
        response=getattr(error,'response',None)
        if not isinstance(response,dict):raise
        code=response.get('Error',{}).get('Code','')
        text=str(error)
        if 'voc-cancel-cred' in text:
            raise DeployError('AWS refused the request: these Learner Lab keys were cancelled (the lab session ended or was restarted).\n'
                              'Fix: in AWS Academy press Start Lab, wait for the green dot, open AWS Details > AWS CLI: Show,\n'
                              'paste all three new lines into .env.aws, then run the same make command again.')
        if code in ('ExpiredToken','ExpiredTokenException','InvalidClientTokenId','UnrecognizedClientException','RequestExpired'):
            raise DeployError('AWS says the keys in .env.aws have expired. Start the lab again and paste the three new key lines into .env.aws.')
        if 'AccessDenied' in code or 'UnauthorizedOperation' in code or 'not authorized' in text:
            operation=getattr(error,'operation_name','an AWS action')
            raise DeployError(f'AWS denied {operation} for this account ({code}). If you use Learner Lab, refresh the keys first; '
                              'if it still fails, this action is not allowed in the account.')
        raise


if __name__=='__main__':run_cli()
