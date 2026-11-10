#!/usr/bin/env python3
"""Regression for native --enable-dynamic-loading with a real model and State.

Start a dynamic server first; --model selects an entry returned by /v1/models.
Only uploaded files created by this suite are deleted.
"""
import argparse
from pathlib import Path

from http_e2e_test import Suite, require


class DynamicSuite(Suite):
    def initial_status(self):
        status = self.call('/v1/server/status')
        assert status['status'] == 'running' and status['dynamic_loading'], status

    def final_status(self):
        self.initial_status()
        assert self.call('/v1/models')['loaded'] == self.args.model

    def run(self):
        self.check('dynamic server', self.initial_status)
        self.check('explicit model load', lambda: require(
            self.call('/v1/model/load', {'model': self.args.model})['loaded']))
        self.check('model selected', self.final_status)
        self.check('token count', lambda: require(self.call('/v1/tokens/count', {'text':'Hello'})['tokens'] > 0))
        for method in ['GET', 'POST', 'DELETE']:
            self.check('removed adapter route '+method, lambda m=method: self.call('/v1/adapters', method=m, expected=404))
        data, headers = self.multipart({}, [('file', Path(self.args.state).name, Path(self.args.state).read_bytes())])
        sid = self.call('/v1/state/upload', raw=data, headers=headers)['state_id']
        self.states.append(sid)
        self.check('uploaded state listed', lambda: require(any(
            entry['state_id'] == sid for entry in self.call('/v1/state/list')['data'])))
        for path, body, count in [('/v1/chat/completions', self.chat, 1), ('/v1/batch/completions', self.batch, 2)]:
            for extra in [{}, {'state_id': sid}]:
                for stream in [False, True]:
                    self.check(f'{path} state={bool(extra)} stream={stream}',
                               lambda p=path,b=body,c=count,e=extra,s=stream: self.completion(p,b|e,c,s))
            for key, value in [('adapter_id', 'old'), ('adapter_version', 'v1'), ('adapter_scale', 1.0)]:
                self.check('removed '+key+' '+path, lambda p=path,b=body,k=key,v=value: require(
                    'no longer supported' in str(self.call(p,b|{k:v},expected=400))))
            self.check('missing state '+path, lambda p=path,b=body: self.call(p,b|{'state_id':self.prefix},expected=400))
        self.check('cached state batch repeats', lambda: [self.completion('/v1/batch/completions', self.batch|{'state_id':sid},2) for _ in range(3)])
        self.check('mixed base/state continuous requests', lambda: self.concurrent_mixed(sid))
        self.check('State header stream', lambda: self.header_stream(sid))
        self.check('model remains selected', self.final_status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8062')
    parser.add_argument('--model', required=True)
    parser.add_argument('--state', default='./rwkv-state-agentic.pth')
    parser.add_argument('--password', default='')
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--output', default='build/adapter-removal-dynamic-e2e')
    args = parser.parse_args()
    suite = DynamicSuite(args)
    try:
        suite.run()
    finally:
        suite.cleanup()
        suite.save()
    print('Report:', suite.output/'report.json')
    return int(any(row['status'] == 'FAIL' for row in suite.results))


if __name__ == '__main__':
    raise SystemExit(main())
