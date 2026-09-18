"""Exercise adapter HTTP lifecycle against fixtures from check_miss.py."""
import argparse
import json
import pathlib
import socket
import subprocess
import time
import urllib.request
import zipfile
import hashlib
import shutil

parser=argparse.ArgumentParser()
parser.add_argument('--build',type=pathlib.Path,required=True)
parser.add_argument('--work',type=pathlib.Path,required=True)
a=parser.parse_args()
with socket.socket() as sock:
    sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
log=(a.work/'http.log').open('w')
p=subprocess.Popen([a.build/'rwkv_lighting_cuda','--model-path',a.work/'tiny.pth',
    '--vocab-path',a.work/'vocab.txt','--port',str(port),'--wkv32','--cmix-sparse','off',
    '--state-db-path',a.work/'http-sessions.db'],stdout=log,stderr=subprocess.STDOUT)
def request(path,data=None,method=None):
    r=urllib.request.Request(f'http://127.0.0.1:{port}'+path,
        data=None if data is None else json.dumps(data).encode(),
        headers={'Content-Type':'application/json'},method=method)
    try:
        with urllib.request.urlopen(r,timeout=15) as response:return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode()) from error
def upload(adapter_id, path, metadata=None):
    boundary='rwkv-miss-test-boundary'
    parts=[f'--{boundary}\r\nContent-Disposition: form-data; name="adapter_id"\r\n\r\n{adapter_id}\r\n'.encode()]
    for file in [path]+([metadata] if metadata else []):
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{file.name}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode()+file.read_bytes()+b'\r\n')
    parts.append(f'--{boundary}--\r\n'.encode())
    req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/adapters',data=b''.join(parts),headers={'Content-Type':f'multipart/form-data; boundary={boundary}'})
    with urllib.request.urlopen(req,timeout=30) as response:return json.load(response)
try:
    for _ in range(100):
        try:request('/v1/adapters');break
        except OSError:
            if p.poll() is not None:raise RuntimeError('server exited; see http.log')
            time.sleep(.05)
    else:raise RuntimeError('server not ready')
    version=request('/v1/adapters',{'adapter_id':'test','path':str(a.work/'learn/adapter-final.pth')})['version']
    # A copied checkpoint must remain independently uploadable without JSON.
    isolated=a.work/'standalone-training.pth'
    shutil.copyfile(a.work/'learn/checkpoint-16/training.pth',isolated)
    assert upload('checkpoint',isolated)['version']==version
    assert upload('final',a.work/'learn/adapter-final.pth')['version']==version
    # Compatibility for checkpoints written before embedded metadata existed.
    legacy=a.work/'legacy-checkpoint';legacy.mkdir(exist_ok=True)
    with zipfile.ZipFile(isolated) as src, zipfile.ZipFile(legacy/'training.pth','w') as dst:
        for item in src.infolist():
            if item.filename!='archive/miss.json':dst.writestr(item,src.read(item.filename))
    meta=json.loads((a.work/'learn/checkpoint-16/checkpoint.json').read_text())
    meta['tensor_digest']=hashlib.sha256((legacy/'training.pth').read_bytes()).hexdigest()
    (legacy/'checkpoint.json').write_text(json.dumps(meta))
    assert request('/v1/adapters',{'adapter_id':'legacy','path':str(legacy/'training.pth')})['version']==version
    assert upload('legacy-upload',legacy/'training.pth',legacy/'checkpoint.json')['version']==version
    stats=request('/v1/adapters')
    assert stats['uploads']==0
    assert stats['gpu_bytes']==0
    # All formats/IDs deduplicate to the same FP16 D payload; no Adam RAM cache.
    with zipfile.ZipFile(a.work/'learn/adapter-final.pth') as z:
        manifest=json.loads(z.read('archive/miss.json'))
    assert stats['ram_bytes']==sum(t['shape'][0]*t['shape'][1]*2 for t in manifest['targets'])
    corrupt=a.work/'corrupt-adapter.pth'
    with zipfile.ZipFile(a.work/'learn/adapter-final.pth') as src, zipfile.ZipFile(corrupt,'w') as dst:
        for item in src.infolist():
            payload=src.read(item.filename)
            if item.filename=='archive/miss.json':
                m=json.loads(payload);m['scale']=123.0;payload=json.dumps(m).encode()
            dst.writestr(item,payload)
    try:
        upload('corrupt',corrupt)
        raise AssertionError('corrupt manifest accepted')
    except urllib.error.HTTPError as error:
        assert error.code==400
    assert request('/v1/adapters')['ram_bytes']==stats['ram_bytes']
    data={'contents':['abcabcabc'],'adapter_id':'test','adapter_version':version,
          'max_tokens':3,'temperature':.001,'top_k':1,'stop_tokens':[]}
    first=request('/v1/batch/completions',data)
    assert request('/v1/adapters')['uploads']==1
    second=request('/v1/batch/completions',data)
    assert request('/v1/adapters')['uploads']==1
    assert first['choices']==second['choices']
    for adapter_id in ['checkpoint','final','legacy','legacy-upload']:
        assert request('/v1/batch/completions',dict(data,adapter_id=adapter_id))['choices']==first['choices']
    assert request('/v1/adapters')['uploads']==1
    # A session can retain independent states for the same adapter at two scales.
    session=dict(data,session_id='miss-session',max_tokens=1)
    request('/state/chat/completions',session)
    request('/state/chat/completions',dict(session,adapter_scale=0))
    entries=request('/state/status',{})['sessions']
    assert len(entries)>=2,entries
    assert request('/state/delete',{'session_id':'miss-session'})['status']=='success'
    request('/v1/adapters',{'adapter_id':'test'},'DELETE')
    for adapter_id in ['checkpoint','final','legacy','legacy-upload']:
        request('/v1/adapters',{'adapter_id':adapter_id},'DELETE')
    assert request('/v1/adapters')['data']==[]
    print('HTTP registration/list/cold+hot generation/deletion passed')
finally:
    p.terminate()
    try:p.wait(timeout=5)
    except subprocess.TimeoutExpired:p.kill();p.wait()
    log.close()
