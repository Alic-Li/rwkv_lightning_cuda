#!/usr/bin/env python3
"""Live HTTP E2E tests. Uses only Python's standard library.

python test/http_e2e_test.py --state ./latest-state.pth --adapter ./adapter-final.pth
The server must already be running. Only resources created by this run are removed.
Raw HTTP responses and report.json are saved in --output (default: build/http-e2e-<id>).
"""
import argparse
import concurrent.futures
import json
import http.client
import os
from pathlib import Path
import threading
import time
import traceback
import urllib.parse
import uuid


class Suite:
    def __init__(self, args):
        self.args = args
        self.prefix = 'http-e2e-' + uuid.uuid4().hex[:12]
        self.output = Path(args.output or 'build/' + self.prefix)
        self.output.mkdir(parents=True, exist_ok=True)
        self.results, self.sessions, self.states, self.adapters = [], [], [], []
        self.lock = threading.Lock()
        self.counter = 0
        self.initial_active = set()
        self.options = dict(max_tokens=8, temperature=1, top_k=1, top_p=0,
                            alpha_presence=0, alpha_frequency=0, stop_tokens=[], chunk_size=1)
        self.chat = self.options | {'messages': [{'role': 'user', 'content': 'Say hello in Chinese.'}], 'think_type': 'none'}
        self.batch = self.options | {'contents': ['English: Hello\n\nChinese:', 'English: Good morning\n\nChinese:']}

    def call(self, path, body=None, method=None, expected=200, headers=None, raw=None, on_event=None):
        method = method or ('POST' if body is not None or raw is not None else 'GET')
        request_headers = dict(headers or {})
        if self.args.password:
            request_headers['Authorization'] = 'Bearer ' + self.args.password
        data = raw
        if body is not None:
            data = json.dumps(body).encode()
            request_headers['Content-Type'] = 'application/json'
        with self.lock:
            self.counter += 1
            number = self.counter
        record = dict(number=number, method=method, path=path, request=body,
                      request_bytes=len(data or b''))
        start = time.monotonic()
        url = urllib.parse.urlsplit(self.args.url)
        connection_type = http.client.HTTPSConnection if url.scheme == 'https' else http.client.HTTPConnection
        connection = connection_type(url.hostname, url.port, timeout=self.args.timeout)
        chunks, events = [], []
        try:
            connection.request(method, url.path.rstrip('/') + path, body=data, headers=request_headers)
            response = connection.getresponse()
            with response:
                record.update(status=response.status, headers=dict(response.headers))
                if 'text/event-stream' in response.headers.get('Content-Type', ''):
                    for line in response:
                        chunks.append(line)
                        if line.startswith(b'data: '):
                            payload = line[6:].strip()
                            event = '[DONE]' if payload == b'[DONE]' else json.loads(payload)
                            events.append(event)
                            if on_event:
                                on_event(event)
                    result = events
                else:
                    chunks.append(response.read())
                    result = json.loads(chunks[0]) if chunks[0] and 'json' in response.headers.get('Content-Type', '') else chunks[0].decode(errors='replace')
            record['response'] = result
            allowed = expected if isinstance(expected, tuple) else (expected,)
            assert record['status'] in allowed, (method, path, record['status'], result)
            return result
        finally:
            connection.close()
            record['seconds'] = round(time.monotonic() - start, 4)
            (self.output / f'{number:04d}-response.txt').write_bytes(b''.join(chunks))
            (self.output / f'{number:04d}-http.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))

    def check(self, name, fn):
        start = time.monotonic()
        try:
            fn()
            row = dict(name=name, status='PASS')
        except Exception as error:
            row = dict(name=name, status='FAIL', error=str(error), traceback=traceback.format_exc())
        row['seconds'] = round(time.monotonic() - start, 3)
        self.results.append(row)
        print(row['status'], name, row.get('error', ''), flush=True)
        self.save()

    def save(self):
        report = dict(base_url=self.args.url, run_id=self.prefix, requests=self.counter,
                      passed=sum(r['status'] == 'PASS' for r in self.results),
                      failed=sum(r['status'] == 'FAIL' for r in self.results), tests=self.results)
        (self.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))

    def multipart(self, fields, files):
        boundary = 'rwkv-' + uuid.uuid4().hex
        data = bytearray()
        for name, value in fields.items():
            data.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
        for name, filename, content in files:
            data.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode())
            data.extend(content)
            data.extend(b'\r\n')
        data.extend(f'--{boundary}--\r\n'.encode())
        return bytes(data), {'Content-Type': 'multipart/form-data; boundary=' + boundary}

    def completion(self, path, body, choices=1, stream=False, on_event=None):
        result = self.call(path, body | {'stream': stream}, on_event=on_event)
        if stream:
            assert result and result[-1] == '[DONE]', result
            assert result.count('[DONE]') == 1, result
            content = [''] * choices
            finishes = [0] * choices
            for event in result[:-1]:
                assert isinstance(event, dict) and 'error' not in event, event
                assert event['object'] == 'chat.completion.chunk', event
                for choice in event['choices']:
                    idx = choice['index']
                    assert 0 <= idx < choices, choice
                    assert not finishes[idx], 'event after finish'
                    content[idx] += choice['delta'].get('content', '')
                    if choice.get('finish_reason') is not None:
                        assert choice['finish_reason'] in ('length', 'stop'), choice
                        finishes[idx] += 1
            assert finishes == [1] * choices, finishes
            assert all(content), content
            return content
        assert result['object'] == 'chat.completion', result
        assert len(result['choices']) == choices, result
        for idx, choice in enumerate(result['choices']):
            assert choice['index'] == idx and choice['message']['role'] == 'assistant', choice
            assert choice['finish_reason'] in ('stop', 'length'), choice
            assert choice['message']['content'], choice
        if path == '/v1/chat/completions':
            usage = result['usage']
            assert usage['prompt_tokens'] > 0 and usage['completion_tokens'] > 0, usage
            assert usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens'], usage
        return [c['message']['content'] for c in result['choices']]

    def session(self, suffix):
        name = self.prefix + '-' + suffix
        self.sessions.append(name)
        return name

    def matrix(self, label, extra):
        for path, body, count in [('/v1/chat/completions', self.chat, 1), ('/v1/batch/completions', self.batch, 2)]:
            outputs = {}
            for streaming in (False, True):
                self.check(f'{label} {path} stream={streaming}',
                           lambda p=path,b=body,c=count,s=streaming: outputs.update({s:self.completion(p,b|extra,c,s)}))
            if len(outputs) == 2:
                self.check(f'{label} {path} deterministic stream parity', lambda: require(outputs[False] == outputs[True], outputs))
        session = self.session(label)
        body = self.options | {'contents': ['User: Hello.\n\nAssistant:'], 'session_id': session} | extra
        self.check(label + ' stateful initial', lambda: self.completion('/state/chat/completions', body))
        # Omit state_id on the continuation to exercise the cached state.
        continuation = {k:v for k,v in body.items() if k != 'state_id'}
        continuation['contents'] = ['\n\nUser: Continue.\n\nAssistant:']
        self.check(label + ' stateful continuation SSE', lambda: self.completion('/state/chat/completions', continuation, stream=True))
        self.check(label + ' session cached', lambda: require(any(e['session_id'].startswith(session + ':') for e in self.call('/state/status', {})['sessions'])))

    def controlled(self, label, action, extra):
        seen = threading.Event()
        active_id = []
        def observe(event):
            if isinstance(event, dict) and any(c.get('delta', {}).get('content') for c in event.get('choices', [])):
                if not active_id:
                    active_id.append(event['id'])
                seen.set()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            future = pool.submit(self.completion, '/v1/chat/completions', self.chat | extra | {'max_tokens': 2048}, 1, True, observe)
            try:
                assert seen.wait(30), 'no stream content before control request'
                rid = active_id[0]
                status = self.call('/v1/server/status')
                assert any(e['id'] == rid for e in status['active_requests']), status
                response = self.call('/v1/server/' + action, {'request_id': rid})
                assert response['stopped' if action == 'stop' else 'paused'], response
                future.result(timeout=30)
                if action == 'pause':
                    status = self.call('/v1/server/status')
                    assert any(e['id'] == rid for e in status['paused_requests']), status
                    self.completion('/v1/server/resume', {'request_id':rid, 'max_tokens':8, 'chunk_size':1}, stream=True)
                    assert not any(e['id'] == rid for e in self.call('/v1/server/status')['paused_requests'])
                    self.call('/state/delete', {'session_id': rid}, expected=(200,404))
            finally:
                if not future.done() and active_id:
                    self.call('/v1/server/stop', {'request_id':active_id[0]})

    def run(self):
        self.check('server status and initial snapshot', self.initial_status)
        self.check('model list', lambda: require(self.call('/v1/models')['data']))
        self.check('CORS response headers', lambda: self.cors())
        paths=['/v1/models','/v1/chat/completions','/v1/batch/completions','/v1/tokens/count','/translate/v1/batch-translate','/state/chat/completions','/state/status','/state/delete','/v1/server/status','/v1/server/stop','/v1/server/pause','/v1/server/resume','/v1/state/upload','/v1/state/list','/v1/state/delete','/v1/adapters']
        for path in paths:
            self.check('OPTIONS ' + path, lambda p=path: self.call(p, method='OPTIONS', expected=(200,204)))
        self.check('adapter browser CORS preflight', self.adapter_preflight)
        for body in [{'text':'你好，RWKV!'}, {'messages':[{'role':'user','content':[{'type':'text','text':'你好'}]}]}, {'contents':['hello','world']}]:
            self.check('token count ' + next(iter(body)), lambda b=body: require(self.call('/v1/tokens/count', b)['tokens'] > 0))
        self.matrix('base', {})
        for think in ['none','fast','free','preferChinese','en','enShort','enLong']:
            self.check('thinking ' + think, lambda t=think: self.completion('/v1/chat/completions', self.chat | {'think_type':t, 'max_tokens':2}))
        self.check('batch text stops', self.batch_text_stops)
        self.check('other endpoint text stops', self.other_text_stops)
        self.check('batch metrics', lambda: self.completion('/v1/batch/completions', self.batch | {'metrics': True},2))
        self.check('batch metrics status', self.metrics)
        self.check('translation', lambda: require(len(self.call('/translate/v1/batch-translate', {'source_lang':'English','target_lang':'Chinese','text_list':['Hello','Good morning']})['translations']) == 2))
        sid = None
        def upload_state():
            nonlocal sid
            data, headers = self.multipart({}, [('file',Path(self.args.state).name,Path(self.args.state).read_bytes())])
            result = self.call('/v1/state/upload', raw=data, headers=headers)
            sid = result['state_id']; self.states.append(sid)
            assert result['size_bytes'] == Path(self.args.state).stat().st_size, result
            assert result['layers'] > 0 and result['heads'] > 0 and result['head_size'] == 64, result
        self.check('upload latest-state.pth', upload_state)
        if sid:
            for method in ['GET','POST']:
                self.check('state list ' + method, lambda m=method: require(any(e['state_id']==sid for e in self.call('/v1/state/list', {} if m=='POST' else None)['data'])))
            self.matrix('loaded-state', {'state_id':sid})
            self.check('state header chat', lambda: require(self.call('/v1/chat/completions', self.chat, headers={'X-RWKV-State-Id':sid})['choices']))
            self.check('state header batch SSE', lambda: self.header_stream(sid))
            self.check('state body/header identical', lambda: require(self.call('/v1/chat/completions',self.chat|{'state_id':sid},headers={'X-RWKV-State-Id':sid})['choices']))
            self.check('translation loaded state', lambda: require(len(self.call('/translate/v1/batch-translate',{'source_lang':'English','target_lang':'Chinese','text_list':['Hi'],'state_id':sid})['translations'])==1))
        aid = self.prefix + '-adapter'; version = None
        def register_adapter():
            nonlocal version
            result = self.call('/v1/adapters', {'adapter_id':aid,'path':str(Path(self.args.adapter).resolve())})
            version = result['version']; self.adapters.append(aid)
            assert len(version)==64,result
        self.check('register adapter-final.pth by path', register_adapter)
        if version:
            self.check('adapter listed with manifest', lambda: require(any(e['id']==aid and e['version']==version and e['manifest']['method']=='miss' for e in self.call('/v1/adapters')['data'])))
            self.matrix('adapter', {'adapter_id':aid,'adapter_version':version})
            self.check('adapter scale override', lambda: self.completion('/v1/chat/completions',self.chat|{'adapter_id':aid,'adapter_scale':0.5}))
            if sid:
                self.matrix('state-and-adapter', {'state_id':sid,'adapter_id':aid})
            self.check('adapter cache GPU and RAM hits', lambda: self.adapter_metrics())
            self.check('adapter pause/resume', lambda: self.controlled('adapter','pause',{'adapter_id':aid}))
        upload_aid = self.prefix + '-multipart'
        def upload_adapter():
            data,headers=self.multipart({'adapter_id':upload_aid},[('file',Path(self.args.adapter).name,Path(self.args.adapter).read_bytes())])
            result=self.call('/v1/adapters',raw=data,headers=headers); self.adapters.append(upload_aid)
            assert result['adapter_id']==upload_aid and len(result['version'])==64,result
        self.check('multipart adapter upload',upload_adapter)
        if upload_aid in self.adapters:
            self.check('multipart adapter inference SSE',lambda: self.completion('/v1/chat/completions',self.chat|{'adapter_id':upload_aid},stream=True))
        self.negative_cases()
        self.check('stream stop by request_id',lambda: self.controlled('base','stop',{}))
        self.check('stream pause and resume by request_id',lambda: self.controlled('base','pause',{}))
        if sid and version:
            self.check('state+adapter pause/resume',lambda: self.controlled('combined','pause',{'state_id':sid,'adapter_id':aid}))
        self.check('concurrent generation requests',self.concurrent)
        if sid and version:
            self.check('continuous mixed state/adapter/scale requests',
                       lambda: self.concurrent_mixed(sid, aid))
        self.check('session header lifecycle',self.session_header)
        for path,body in [('/v1/chat/completions',self.chat),('/v1/batch/completions',self.batch),('/state/chat/completions',self.options|{'session_id':self.session('connection-close'),'contents':['User: Hello.\n\nAssistant:']})]:
            self.check('SSE Connection: close '+path,lambda p=path,b=body:self.connection_close(p,b))

    def adapter_preflight(self):
        origin = 'http://localhost:5173'
        for method in ('GET', 'POST', 'DELETE'):
            self.call('/v1/adapters', method='OPTIONS', expected=(200, 204), headers={
                'Origin': origin,
                'Access-Control-Request-Method': method,
                'Access-Control-Request-Headers': 'content-type,authorization',
            })
            record = json.loads((self.output / f'{self.counter:04d}-http.json').read_text())
            headers = {k.lower(): v for k, v in record['headers'].items()}
            assert headers['access-control-allow-origin'] in ('*', origin), headers
            methods = {v.strip().upper() for v in headers['access-control-allow-methods'].split(',')}
            assert method in methods, headers
            allowed = {v.strip().lower() for v in headers['access-control-allow-headers'].split(',')}
            assert {'content-type', 'authorization'} <= allowed, headers

    def cors(self):
        self.call('/v1/server/status')
        r=json.loads((self.output/f'{self.counter:04d}-http.json').read_text())
        h={k.lower():v for k,v in r['headers'].items()}
        assert h['access-control-allow-origin']=='*',h
        assert 'X-RWKV-State-Id' in h['access-control-allow-headers'],h

    def metrics(self):
        last=self.call('/v1/server/status')['last_request']
        assert last['endpoint']=='batch.completions' and last['prefill_progress']==1,last
        assert last['prefill_speed']>0 and last['decode_speed']>0,last

    def adapter_metrics(self):
        r=self.call('/v1/adapters')
        assert r['ram_hits']>0 and r['gpu_hits']>0 and r['gpu_misses']>0,r
        assert r['gpu_peak_bytes']>0 and r['ram_bytes']>0,r

    def batch_text_stops(self):
        path = '/v1/batch/completions'
        body = {k: v for k, v in self.batch.items() if k != 'stop_tokens'}
        for invalid in [None, 'stop', [''], [1], [None]]:
            self.call(path, body | {'stop': invalid}, expected=400)
        for legacy in [[], None, [0]]:
            self.call(path, body | {'stop': [], 'stop_tokens': legacy}, expected=400)
        # Greedy baseline supplies exact generated text ending at a token boundary.
        body |= {'contents': [self.batch['contents'][0]], 'max_tokens': 3, 'stop': []}
        baseline = self.call(path, body)['choices'][0]['message']['content']
        assert baseline, 'expected non-empty greedy baseline'
        body |= {'max_tokens': 16, 'stop': [baseline], 'chunk_size': 100}
        result = self.call(path, body)['choices'][0]
        assert result['message']['content'] == baseline, result
        assert result['finish_reason'] == 'stop' and result['matched_stop'] == baseline, result
        events = self.call(path, body | {'stream': True})
        assert events[-1] == '[DONE]', events
        choices = [c for event in events[:-1] for c in event['choices']]
        assert ''.join(c['delta'].get('content', '') for c in choices) == baseline, choices
        finishes = [c for c in choices if c.get('finish_reason') is not None]
        assert len(finishes) == 1, finishes
        assert finishes[0]['finish_reason'] == 'stop', finishes
        assert finishes[0]['matched_stop'] == baseline, finishes

    def other_text_stops(self):
        cases = [
            ('/v1/chat/completions', self.chat),
            ('/state/chat/completions', self.options | {'contents': ['User: Hello.\n\nAssistant:']}),
            ('/translate/v1/batch-translate', self.options | {
                'source_lang': 'English', 'target_lang': 'Chinese', 'text_list': ['Hello']}),
        ]
        for path, original in cases:
            body = {k: v for k, v in original.items() if k != 'stop_tokens'}
            stateful = path.startswith('/state/')
            translation = path.startswith('/translate/')
            def request(extra):
                payload = body | extra
                if stateful:
                    payload['session_id'] = self.session('text-stop-' + uuid.uuid4().hex[:8])
                return payload
            for invalid in [None, 'stop', [''], [1]]:
                self.call(path, request({'stop': invalid}), expected=400)
            self.call(path, request({'stop': [], 'stop_tokens': []}), expected=400)
            baseline = self.call(path, request({'max_tokens': 3, 'stop': []}))
            baseline = (baseline['translations'][0]['text'] if translation else
                        baseline['choices'][0]['message']['content'])
            assert baseline, (path, baseline)
            extra = {'max_tokens': 16, 'stop': [baseline], 'chunk_size': 100}
            result = self.call(path, request(extra))
            choice = result['translations'][0] if translation else result['choices'][0]
            assert choice['matched_stop'] == baseline and choice['finish_reason'] == 'stop', choice
            assert (choice['text'] if translation else choice['message']['content']) == baseline, choice
            if not translation:
                events = self.call(path, request(extra | {'stream': True}))
                assert events[-1] == '[DONE]', events
                choices = [c for e in events[:-1] for c in e['choices']]
                assert ''.join(c['delta'].get('content', '') for c in choices) == baseline, choices
                finishes = [c for c in choices if c.get('finish_reason') is not None]
                assert len(finishes) == 1 and finishes[0]['matched_stop'] == baseline, finishes

    def header_stream(self,sid):
        events=self.call('/v1/batch/completions',self.batch|{'stream':True},headers={'X-RWKV-State-Id':sid})
        assert events[-1]=='[DONE]' and not any(isinstance(e,dict) and 'error' in e for e in events),events

    def negative_cases(self):
        cases=[('/v1/tokens/count',{},400),('/v1/batch/completions',{'contents':[]},400),
               ('/translate/v1/batch-translate',{'text_list':['Hi']},400),
               ('/translate/v1/batch-translate',{'target_lang':'Chinese','text_list':[]},400),
               ('/state/chat/completions',{'contents':['Hi']},400),
               ('/state/chat/completions',{'session_id':self.prefix,'contents':['a','b']},400),
               ('/state/delete',{},400),('/state/delete',{'session_id':self.prefix+'-missing'},404),
               ('/v1/state/delete',{},400),('/v1/state/delete',{'state_id':self.prefix+'-missing'},404),
               ('/v1/server/resume',{},400),('/v1/server/resume',{'request_id':self.prefix+'-missing'},404)]
        for path,body,status in cases:
            self.check('negative '+path+' '+str(body),lambda p=path,b=body,s=status: require('error' in (r:=self.call(p,b,expected=s)) or p=='/state/delete' or (p=='/v1/state/delete' and s==404 and r.get('deleted') is False)))
        for path,body in [('/v1/chat/completions',self.chat),('/v1/batch/completions',self.batch),('/state/chat/completions',self.options|{'session_id':self.session('errors'),'contents':['hello']})]:
            for extra in [{'state_id':self.prefix+'-missing'},{'state_id':123},{'adapter_id':self.prefix+'-missing'},{'adapter_scale':'invalid'},{'adapter_version':'missing'}]:
                self.check('negative '+path+' '+str(extra),lambda p=path,b=body,e=extra: require('error' in self.call(p,b|e,expected=400)))
            self.check('invalid JSON '+path,lambda p=path: self.call(p,raw=b'{bad json',headers={'Content-Type':'application/json'},expected=400))
        data,headers=self.multipart({},[('file','bad.pth',b'invalid pth')])
        self.check('invalid state upload',lambda:self.call('/v1/state/upload',raw=data,headers=headers,expected=400))
        data,headers=self.multipart({},[])
        self.check('empty state upload',lambda:self.call('/v1/state/upload',raw=data,headers=headers,expected=400))
        data,headers=self.multipart({},[('file','bad.pth',b'invalid pth')])
        self.check('adapter upload missing adapter_id',lambda:self.call('/v1/adapters',raw=data,headers=headers,expected=400))
        self.check('adapter nonexistent path',lambda:self.call('/v1/adapters',{'adapter_id':self.prefix+'-bad','path':'/nonexistent/e2e-adapter.pth'},expected=400))
        for action,key in [('stop','stopped'),('pause','paused')]:
            self.check(action+' unknown request',lambda a=action,k=key:require(self.call('/v1/server/'+a,{'request_id':self.prefix+'-missing'})[k] is False))

    def concurrent(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(self.completion,'/v1/chat/completions',self.chat|{'messages':[{'role':'user','content':f'Count starting at {i}.'}]},1,bool(i%2)) for i in range(4)]
            for f in futures: f.result(timeout=self.args.timeout)

    def concurrent_mixed(self, state_id, adapter_id):
        barrier = threading.Barrier(8)
        def run(i):
            body = self.chat | dict(max_tokens=16, temperature=0.5+i*0.1,
                                    alpha_presence=i*0.2, alpha_frequency=i*0.03,
                                    alpha_decay=0.9+i*0.01)
            if i & 1:
                body['state_id'] = state_id
            if i & 2:
                body.update(adapter_id=adapter_id, adapter_scale=0.5 if i & 4 else 1.0)
            barrier.wait()
            return self.completion('/v1/chat/completions', body, 1, bool(i & 4))
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(run, i) for i in range(8)]
            for future in futures:
                future.result(timeout=self.args.timeout)

    def session_header(self):
        session=self.session('header')
        body=self.options|{'contents':['User: hello\n\nAssistant:']}
        require(self.call('/state/chat/completions',body,headers={'X-RWKV-Session-Id':session})['choices'])
        require(self.call('/state/chat/completions',body|{'session_id':session},headers={'X-RWKV-Session-Id':session})['choices'])
        self.call('/state/chat/completions',body|{'session_id':session},headers={'X-RWKV-Session-Id':session+'-other'},expected=400)
        self.call('/state/delete',{},headers={'X-RWKV-Session-Id':session})

    def cleanup(self):
        for session in self.sessions:
            self.check('cleanup session '+session,lambda s=session:self.call('/state/delete',{'session_id':s},expected=(200,404)))
        for sid in self.states:
            self.check('cleanup uploaded state '+sid,lambda s=sid:require(self.call('/v1/state/delete?'+urllib.parse.urlencode({'state_id':s}),method='DELETE')['deleted']))
            self.check('deleted state absent',lambda s=sid:require(not any(e['state_id']==s for e in self.call('/v1/state/list')['data'])))
        for aid in self.adapters:
            self.check('cleanup adapter '+aid,lambda a=aid:require(self.call('/v1/adapters',{'adapter_id':a},method='DELETE')['deleted']))
        self.check('test adapters absent',lambda:require(not any(e['id'].startswith(self.prefix) for e in self.call('/v1/adapters')['data'])))
        self.check('final server has no newly leaked active or paused requests',lambda:self.final_status())

    def initial_status(self):
        status=self.call('/v1/server/status')
        self.initial_active={r['id'] for r in status['active_requests']}
        (self.output/'initial-status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
        assert status['status']=='running' and status['prefill_queue']['reserved_bsz']==0,status

    def connection_close(self,path,body):
        events=self.call(path,body|{'stream':True},headers={'Connection':'close'})
        assert events and events[-1]=='[DONE]', 'HTTP 200 but empty/incomplete SSE body with Connection: close'
        assert not any(isinstance(e,dict) and 'error' in e for e in events),events

    def final_status(self):
        status=self.call('/v1/server/status')
        new_active=[r for r in status['active_requests'] if r['id'] not in self.initial_active]
        assert status['status']=='running' and not new_active, {'new_leaked_active_requests':new_active}
        assert not status['paused_requests'],status
        assert status['prefill_queue']['reserved_bsz']==0,status


def require(value, detail=None):
    assert value, detail


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default=os.environ.get('BASE_URL','http://127.0.0.1:8000'))
    parser.add_argument('--password',default=os.environ.get('PASSWORD',''))
    parser.add_argument('--state',default='./latest-state.pth')
    parser.add_argument('--adapter',default='./adapter-final.pth')
    parser.add_argument('--timeout',type=float,default=120)
    parser.add_argument('--output')
    args=parser.parse_args()
    for name in [args.state,args.adapter]:
        if not Path(name).is_file(): parser.error('file does not exist: '+name)
    suite=Suite(args)
    try: suite.run()
    finally: suite.cleanup(); suite.save()
    print('Report:',suite.output/'report.json',flush=True)
    return int(any(r['status']=='FAIL' for r in suite.results))


if __name__=='__main__':
    raise SystemExit(main())
