"""Shared, offline Alex AFI record contract. No AWS or hardware operations."""
import hashlib
import re

ACCOUNT='438374459156'
REGION='us-east-1'
BUCKET='cg-rachit-afi-438374459156-us-east-1'
BASELINES=frozenset({'agfi-0d06cb48544a02106','agfi-00c2d61ec51bd89aa'})
HASH_FIELDS=('evidence_manifest_sha256','inputs_sha256','dcp_sha256','ip_receipt_sha256',
             'simulation_receipt_sha256','qualification_receipt_sha256','package_sha256','package_receipt_sha256')


def require(ok,message):
    if not ok:raise ValueError(message)

def image_tags(record):
    return {'c1:'+key:record[key] for key in
            ('build_id','inputs_sha256','package_sha256','client_token','evidence_manifest_sha256')}

def token(record):
    return hashlib.sha256((''.join(record[k] for k in
        ('prefix','package_sha256','inputs_sha256','evidence_manifest_sha256'))).encode()).hexdigest()

def make_request(record):
    return dict(InputStorageLocation=dict(Bucket=BUCKET,Key=record['prefix']+'/design.tar'),
        LogsStorageLocation=dict(Bucket=BUCKET,Key=record['prefix']+'/logs'),
        Name='c1-alex-'+record['build_id'],Description='C1 Alex archive SHA256 '+record['package_sha256'],
        ClientToken=record['client_token'],TagSpecifications=[dict(ResourceType='fpga-image',
            Tags=[dict(Key=k,Value=v) for k,v in image_tags(record).items()])])

def validate_identity(record, *, require_available=False):
    """Publisher may validate prepared/pending records; sessions require available."""
    require(type(record.get('schema')) is int and record['schema']==1 and
        record.get('kind')=='c1-alex-fpga-image','Wrong Alex image-record schema')
    require((record.get('account'),record.get('region'),record.get('bucket'))==(ACCOUNT,REGION,BUCKET),
            'Wrong AWS image account/region/bucket')
    source=record.get('source_revision','');build=record.get('build_id','')
    require(isinstance(source,str) and re.fullmatch('[0-9a-f]{40}',source) and build==source[:8],
            'Image hardware source/build identity mismatch')
    for key in HASH_FIELDS:
        require(isinstance(record.get(key),str) and re.fullmatch('[0-9a-f]{64}',record[key]),
                'Missing/invalid deployment hash: '+key)
    require(type(record.get('package_bytes')) is int and record['package_bytes']>0,'Missing package size')
    prefix=record.get('prefix','')
    require(isinstance(prefix,str) and re.fullmatch('c1-alex/'+build+'/[0-9a-f]{32}',prefix),
            'Invalid unique Alex publication prefix')
    require(record.get('client_token')==token(record),'Invalid idempotent publication token')
    require(record.get('publication_request')==make_request(record),'Publication request differs from bound artifact')
    state=record.get('state')
    require(state in ('prepared','pending','available','failed','unavailable'),'Unknown publication state')
    if require_available:require(state=='available','Candidate image is not available')
    if state in ('pending','available','unavailable') or record.get('image_id') or record.get('global_image_id'):
        require(isinstance(record.get('image_id'),str) and re.fullmatch('afi-[0-9a-f]{17}',record['image_id']) and
                isinstance(record.get('global_image_id'),str) and re.fullmatch('agfi-[0-9a-f]{17}',record['global_image_id']) and
                record['global_image_id'] not in BASELINES,'Missing/distinct candidate regional/global image IDs')
    return record

def validate_record(record, verified):
    """State-independent publisher binding; does not authorize a physical session."""
    validate_identity(record)
    # Evidence may be built on the dedicated CPU host. The session independently
    # enforces its sole FPGA runtime target before accessing a device.
    require(type(verified.get('schema')) is int and verified['schema']==1 and verified.get('kind')=='verified-c1-alex-portable-evidence' and
            verified.get('route_qualified') is True and verified.get('hardware_measured') is False and
            verified.get('review_required') is False and
            verified.get('instance_id') in ('i-0c78de2253f66519d', 'i-06cf8792fb8f15534'),'Portable evidence is not qualified')
    expected={'source_revision':verified['source_revision'],'build_id':verified['build_id'],
        'evidence_manifest_sha256':verified['manifest_sha256'],'inputs_sha256':verified['synth_inputs_sha256'],
        'dcp_sha256':verified['final_dcp_sha256'],**{k:verified[k] for k in
            ('ip_receipt_sha256','simulation_receipt_sha256','qualification_receipt_sha256')}}
    require(all(record.get(k)==v for k,v in expected.items()),'Image record differs from qualified evidence')
    return dict(expected,account=ACCOUNT,region=REGION,package_sha256=record['package_sha256'],
        package_bytes=record['package_bytes'],package_receipt_sha256=record['package_receipt_sha256'],
        client_token=record['client_token'],agfi_id=record.get('global_image_id'),
        afi_id=record.get('image_id'),route_qualified=True)


def verify_deployment(record, verified):
    """Strict session wrapper: only an available, distinct candidate may proceed."""
    result=validate_record(record,verified)
    validate_identity(record,require_available=True)
    verify_aws_image(record.get('last_describe'), record)
    return result


def verify_aws_image(response, record):
    """Reconcile a recorded or fresh AWS response with the exact publication."""
    validate_identity(record, require_available=True)
    require(isinstance(response, dict) and isinstance(response.get('FpgaImages'), list)
            and len(response['FpgaImages']) == 1, 'Expected one AWS image response')
    image = response['FpgaImages'][0]
    require(isinstance(image, dict) and
            image.get('FpgaImageId') == record['image_id'] and
            image.get('FpgaImageGlobalId') == record['global_image_id'] and
            image.get('OwnerId') == ACCOUNT and
            isinstance(image.get('State'), dict) and image['State'].get('Code') == 'available' and
            isinstance(image.get('ShellVersion'), str) and image['ShellVersion'].lower() == '0x10212415',
            'AWS image identity, owner, state or shell differs')
    rows = image.get('Tags', [])
    require(isinstance(rows, list) and all(isinstance(row, dict) and
            isinstance(row.get('Key'), str) and isinstance(row.get('Value'), str)
            for row in rows), 'Invalid AWS image tags')
    tags = {row['Key']: row['Value'] for row in rows}
    require(len(tags) == len(rows) and all(tags.get(key) == value
            for key, value in image_tags(record).items()), 'AWS image provenance tags differ')
    return dict(afi_id=record['image_id'], agfi_id=record['global_image_id'], owner=ACCOUNT,
                state=image['State'], shell_version=image['ShellVersion'], tags=tags)
