#!/usr/bin/env python3
"""Live Launcher -> Router -> native workers regression with a real State file.

--url is the Launcher backend prefix, e.g.
http://127.0.0.1:18081/api/v1/backends/formal-router
--router is the Router URL; /state/* remains a Router/native API.
Only files and sessions created by this suite are deleted. No token parity is
required between independent sparse-FFN generations.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from http_e2e_test import Suite, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--router', required=True)
    parser.add_argument('--worker', action='append', required=True)
    parser.add_argument('--state', default='./rwkv-state-agentic.pth')
    parser.add_argument('--password', default='')
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--output', default='build/router-launcher-e2e')
    args = parser.parse_args()
    suite = Suite(args)
    router = Suite(SimpleNamespace(**(vars(args) | {
        'url': args.router, 'output': str(suite.output/'router')})))
    workers = [Suite(SimpleNamespace(**(vars(args) | {
        'url': url, 'output': str(suite.output/f'worker-{i}')})))
        for i, url in enumerate(args.worker)]

    def replicas(sid, present):
        for worker in workers:
            entries = worker.call('/v1/state/list')['data']
            require(any(e['state_id'] == sid for e in entries) == present, entries)

    def affinity(sid):
        session = suite.prefix+'-affinity'
        for _ in range(6):
            suite.completion('/v1/chat/completions', suite.chat | {
                'session_id': session, 'state_id': sid})
        routes = []
        for path in suite.output.glob('*-http.json'):
            record = json.loads(path.read_text())
            if (record.get('request') or {}).get('session_id') == session:
                headers = {k.lower(): v for k,v in record['headers'].items()}
                routes.append(headers.get('x-rwkv-backend'))
        require(len(routes) == 6 and all(routes) and len(set(routes)) == 1, routes)

    def session_cache(sid):
        session = router.session('uploaded-state')
        body = router.options | {'contents':['User: Hello\n\nAssistant:'],
                                 'state_id': sid, 'session_id': session}
        router.completion('/state/chat/completions', body)
        # Continue from cached State without supplying the initial State again.
        body.pop('state_id')
        body['contents'] = ['\n\nUser: Continue\n\nAssistant:']
        router.completion('/state/chat/completions', body, stream=True)
        owners = 0
        for worker in workers:
            entries = worker.call('/state/status', {})['sessions']
            owners += any(e['session_id'].startswith(session+':') for e in entries)
        require(owners == 1, f'Session found on {owners} workers')
        router.call('/state/delete', {'session_id': session})
        router.sessions.remove(session)
        for worker in workers:
            require(not any(e['session_id'].startswith(session+':')
                for e in worker.call('/state/status', {})['sessions']))

    try:
        for i, worker in enumerate(workers):
            suite.check(f'worker {i} ready', worker.initial_status)
        suite.check('Launcher to Router models', lambda: require(suite.call('/v1/models')['data']))
        for method in ['GET', 'POST', 'DELETE', 'OPTIONS']:
            suite.check('retired management '+method, lambda m=method: suite.call('/v1/adapters', method=m, expected=404))
        for path, body in [('/v1/chat/completions', suite.chat), ('/v1/batch/completions', suite.batch)]:
            for key in ['adapter_id', 'adapter_version', 'adapter_scale']:
                suite.check('retired field '+path+' '+key, lambda p=path,b=body,k=key:
                    suite.call(p,b|{k:None},expected=400))
        def upload():
            data, headers = suite.multipart({}, [('file', Path(args.state).name, Path(args.state).read_bytes())])
            sid = suite.call('/v1/state/upload', raw=data, headers=headers)['state_id']
            suite.states.append(sid)
        suite.check('State upload through full chain', upload)
        require(suite.states, 'State upload failed')
        sid = suite.states[0]
        suite.check('State confirmed on every worker', lambda: replicas(sid, True))
        suite.check('Router consistent State list', lambda: require(any(
            e['state_id'] == sid for e in suite.call('/v1/state/list')['data'])))
        for label, extra in [('base', {}), ('State', {'state_id':sid})]:
            for path, body, count in [('/v1/chat/completions',suite.chat,1),('/v1/batch/completions',suite.batch,2)]:
                for stream in [False, True]:
                    suite.check(f'{label} {path} stream={stream}', lambda p=path,b=body,c=count,s=stream,e=extra:
                        suite.completion(p,b|e,c,s))
        suite.check('State header SSE', lambda: suite.header_stream(sid))
        suite.check('State repeated requests', lambda: [suite.completion(
            '/v1/batch/completions',suite.batch|{'state_id':sid},2) for _ in range(4)])
        suite.check('mixed base/State continuous batching', lambda: suite.concurrent_mixed(sid))
        suite.check('State session affinity', lambda: affinity(sid))
        suite.check('State cache continuation and Session deletion', lambda: session_cache(sid))
        suite.check('unknown State has no zero-State fallback', lambda: suite.call(
            '/v1/chat/completions',suite.chat|{'state_id':suite.prefix+'-missing'},expected=503))
    finally:
        router.cleanup()
        suite.cleanup()
        for sid in suite.states:
            suite.check('deleted State absent on every worker', lambda s=sid: replicas(s,False))
            suite.check('deleted State cannot infer', lambda s=sid: suite.call(
                '/v1/chat/completions',suite.chat|{'state_id':s},expected=503))
        for i, worker in enumerate(workers):
            suite.check(f'worker {i} no leaked requests', worker.final_status)
            worker.save()
        suite.results.extend({'name':'Router: '+r['name'], **{k:v for k,v in r.items() if k!='name'}} for r in router.results)
        router.save()
        suite.save()
    print('Report:', suite.output/'report.json')
    return int(any(r['status']=='FAIL' for r in suite.results))


if __name__ == '__main__':
    raise SystemExit(main())
