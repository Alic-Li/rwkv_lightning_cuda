"""Exercise adapter HTTP lifecycle against fixtures from check_miss.py."""
import argparse
import json
import pathlib
import socket
import subprocess
import time
import urllib.request

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
try:
    for _ in range(100):
        try:request('/v1/adapters');break
        except OSError:
            if p.poll() is not None:raise RuntimeError('server exited; see http.log')
            time.sleep(.05)
    else:raise RuntimeError('server not ready')
    version=request('/v1/adapters',{'adapter_id':'test','path':str(a.work/'learn/adapter')})['version']
    assert request('/v1/adapters')['uploads']==0
    data={'contents':['abcabcabc'],'adapter_id':'test','adapter_version':version,
          'max_tokens':3,'temperature':.001,'top_k':1,'stop_tokens':[]}
    first=request('/v1/batch/completions',data)
    assert request('/v1/adapters')['uploads']==1
    second=request('/v1/batch/completions',data)
    assert request('/v1/adapters')['uploads']==1
    assert first['choices']==second['choices']
    # A session can retain independent states for the same adapter at two scales.
    session=dict(data,session_id='miss-session',max_tokens=1)
    request('/state/chat/completions',session)
    request('/state/chat/completions',dict(session,adapter_scale=0))
    entries=request('/state/status',{})['sessions']
    assert len(entries)>=2,entries
    assert request('/state/delete',{'session_id':'miss-session'})['status']=='success'
    request('/v1/adapters',{'adapter_id':'test'},'DELETE')
    assert request('/v1/adapters')['data']==[]
    print('HTTP registration/list/cold+hot generation/deletion passed')
finally:
    p.terminate()
    try:p.wait(timeout=5)
    except subprocess.TimeoutExpired:p.kill();p.wait()
    log.close()
