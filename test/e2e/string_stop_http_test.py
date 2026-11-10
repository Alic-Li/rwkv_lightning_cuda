#!/usr/bin/env python3
"""Live string-stop regression suite; requires a running HTTP backend, no model files.

python test/string_stop_http_test.py --url http://127.0.0.1:8000
Raw HTTP responses and report.json are saved under build/string-stop-http-*.
"""
import argparse
import ast
import concurrent.futures
import json
import os
from pathlib import Path
import time
import uuid

from http_e2e_test import Suite, require


class StringStopSuite(Suite):
    def __init__(self, args):
        super().__init__(args)
        self.options = dict(max_tokens=16, temperature=1, top_k=1, top_p=0,
                            alpha_presence=0, alpha_frequency=0, alpha_decay=1, stop=[])
        self.cases = {
            'chat': ('/v1/chat/completions', {'messages': [{'role': 'user', 'content': 'Say hello in Chinese.'}], 'think_type': 'none'}),
            'batch': ('/v1/batch/completions', {'contents': ['English: Hello\n\nChinese:']}),
            'state': ('/state/chat/completions', {'contents': ['User: Hello.\n\nAssistant:']}),
            'translation': ('/translate/v1/batch-translate', {'source_lang': 'English', 'target_lang': 'Chinese', 'text_list': ['Hello']}),
        }
        self.baselines = {}
        self.owned_requests = set()

    def body(self, kind, extra=None):
        body = self.options | self.cases[kind][1] | (extra or {})
        if extra and 'stop_tokens' in extra and 'stop' not in extra:
            body.pop('stop', None)
        if kind == 'state':
            body['session_id'] = self.session('stop-' + uuid.uuid4().hex[:10])
        return body

    def request(self, kind, extra=None, **kwargs):
        return self.call(self.cases[kind][0], self.body(kind, extra), **kwargs)

    @staticmethod
    def unpack(result, stream=False, translation=False):
        if not stream:
            choices = result['translations' if translation else 'choices']
            text = [c['text'] if translation else c['message']['content'] for c in choices]
            return text, choices
        assert result and result[-1] == '[DONE]' and result.count('[DONE]') == 1, result
        text, finishes, done = {}, {}, set()
        for event in result[:-1]:
            assert 'error' not in event and event['object'] == 'chat.completion.chunk', event
            for choice in event['choices']:
                i = choice['index']
                assert i not in done, ('event after row finish', choice)
                text[i] = text.get(i, '') + choice['delta'].get('content', '')
                if choice.get('finish_reason') is not None:
                    assert choice['delta'] == {}, choice
                    finishes[i] = choice
                    done.add(i)
        assert text.keys() == finishes.keys(), (text, finishes)
        return [text[i] for i in sorted(text)], [finishes[i] for i in sorted(finishes)]

    def positive(self, kind, stop, expected, stream=False, **extra):
        result = self.request(kind, {'stop': stop, 'stream': stream, **extra})
        texts, choices = self.unpack(result, stream, kind == 'translation')
        assert texts == [expected], (kind, texts, expected)
        assert choices[0]['finish_reason'] == 'stop', choices
        assert choices[0]['matched_stop'] == next(s for s in stop if expected.endswith(s)), choices
        return result

    def baseline(self, kind):
        result = self.request(kind, {'max_tokens': 3})
        prefix, choices = self.unpack(result, translation=kind == 'translation')
        assert prefix[0] and choices[0]['finish_reason'] == 'length', result
        result = self.request(kind)
        full, choices = self.unpack(result, translation=kind == 'translation')
        assert full[0].startswith(prefix[0]) and len(full[0]) > len(prefix[0]), (prefix, full)
        assert choices[0]['finish_reason'] == 'length' and 'matched_stop' not in choices[0], choices
        self.baselines[kind] = (prefix[0], full[0])

    def no_match(self, kind, stream, stop):
        expected = self.baselines[kind][1]
        result = self.request(kind, {'stop': stop, 'stream': stream})
        text, choices = self.unpack(result, stream, kind == 'translation')
        assert text == [expected], (text, expected)
        assert choices[0]['finish_reason'] == 'length' and 'matched_stop' not in choices[0], choices

    def legacy(self, kind, stream):
        result = self.request(kind, {'stop_tokens': [], 'stream': stream})
        text, choices = self.unpack(result, stream, kind == 'translation')
        assert text == [self.baselines[kind][1]], text
        assert choices[0]['finish_reason'] == 'length' and 'matched_stop' not in choices[0], choices

    def literal_batch(self, stop, stream):
        prompt = 'User: Use the calculator tool to calculate 1+1.\n\nAssistant: <tool_call>\n{"name":"calculator","arguments":{"expression":"1+1"}}\n'
        extra = {'contents': [prompt], 'max_tokens': 64}
        baseline = self.request('batch', extra)
        text = self.unpack(baseline)[0][0]
        assert stop in text, ('literal not generated by this model', stop, text)
        # Choose the first token boundary whose decoded text ends in the stop.
        events = self.request('batch', extra | {'stream': True, 'chunk_size': 1})
        prefix = ''
        for event in events[:-1]:
            for choice in event['choices']:
                prefix += choice['delta'].get('content', '')
                if prefix.endswith(stop):
                    return self.positive('batch', [stop], prefix, stream,
                                         contents=[prompt], max_tokens=128, chunk_size=100)
        raise AssertionError(('stop never ended at a token boundary', stop, text))

    def independent_batch(self, stream):
        prompts = ['English: Hello\n\nChinese:',
                   'System: You are helpful.\n\nUser: Hello.\n\nAssistant:']
        baseline = self.request('batch', {'contents': prompts, 'max_tokens': 3})
        first = self.unpack(baseline)[0][0]
        baseline = self.request('batch', {'contents': prompts, 'max_tokens': 14})
        outputs = self.unpack(baseline)[0]
        assert outputs[0].startswith(first) and first not in outputs[1], (first, outputs)
        result = self.request('batch', {'contents': prompts, 'max_tokens': 14, 'stop': [first],
                                        'stream': stream, 'chunk_size': 1})
        texts, finishes = self.unpack(result, stream)
        assert texts == [first, outputs[1]], (texts, outputs)
        assert [c['finish_reason'] for c in finishes] == ['stop', 'length'], finishes
        assert finishes[0]['matched_stop'] == first and 'matched_stop' not in finishes[1], finishes
        if stream:
            choices = [c for e in result[:-1] for c in e['choices']]
            early = next(i for i, c in enumerate(choices) if c['index'] == 0 and c.get('finish_reason'))
            assert any(c['index'] == 1 and c['delta'].get('content') for c in choices[early + 1:]), choices

    def prompt_only(self, stream):
        prompt = 'User: Repeat exactly: </tool_call>\n\nAssistant: </tool_'
        base = self.request('batch', {'contents': [prompt], 'max_tokens': 1})
        expected = self.unpack(base)[0][0]
        assert expected and '</tool_' not in expected, expected
        result = self.request('batch', {'contents': [prompt], 'max_tokens': 1,
                                      'stop': ['</tool_'], 'stream': stream})
        texts, choices = self.unpack(result, stream)
        assert texts == [expected] and choices[0]['finish_reason'] == 'length', (texts, choices)

    def pause(self, stop):
        paused_id = []
        initial = []
        def observe(event):
            if not isinstance(event, dict):
                return
            if any(c['delta'].get('content') for c in event.get('choices', [])) and not paused_id:
                rid = event['id']
                self.owned_requests.add(rid)
                paused_id.append(rid)
                result = self.call('/v1/server/pause', {'request_id': rid})
                assert result['paused'], result
        events = self.request('chat', {'max_tokens': 4096, 'stop': stop, 'stream': True,
                                       'chunk_size': 1}, on_event=observe)
        initial = self.unpack(events, True)[0][0]
        assert paused_id, events
        rid = paused_id[0]
        status = self.call('/v1/server/status')
        assert any(r['id'] == rid for r in status['paused_requests']), status
        return rid, initial

    def resume(self, mode):
        # Long deterministic text gives the pause request plenty of time to arrive.
        baseline = self.request('chat', {'max_tokens': 256})
        target = self.unpack(baseline)[0][0]
        assert len(target) > 100, target
        rid, initial = self.pause([target])
        assert target.startswith(initial) and len(initial) < len(target), (initial, target)
        if mode == 'inherit':
            invalids = [{'stop': None}, {'stop': ''}, {'stop': ['']}, {'stop': [1]},
                        {'stop': [], 'stop_tokens': []}, {'stop': [], 'stop_tokens': None}]
            for invalid in invalids:
                self.call('/v1/server/resume', {'request_id': rid, **invalid}, expected=400)
                status = self.call('/v1/server/status')
                assert any(r['id'] == rid for r in status['paused_requests']), status
            extra = {}
        elif mode == 'replace':
            # Full remaining generated text is an exact new stop, independent of old tail.
            extra = {'stop': [target[len(initial):]]}
        else:
            extra = {'stop_tokens': [], 'max_tokens': 4}
        events = self.call('/v1/server/resume', {'request_id': rid, 'chunk_size': 100, **extra})
        texts, choices = self.unpack(events, True)
        if mode == 'legacy':
            assert choices[0]['finish_reason'] == 'length' and 'matched_stop' not in choices[0], choices
        else:
            assert initial + texts[0] == target, (initial, texts, target)
            assert choices[0]['finish_reason'] == 'stop', choices
            assert choices[0]['matched_stop'] == (target if mode == 'inherit' else extra['stop'][0]), choices
        self.call('/state/delete', {'session_id': rid}, expected=(200, 404))

    def concurrent_stops(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            jobs = []
            for kind in self.cases:
                prefix = self.baselines[kind][0]
                jobs.append(pool.submit(self.positive, kind, [prefix], prefix, kind != 'translation', chunk_size=100))
            for job in jobs:
                job.result(timeout=self.args.timeout)

    def run(self):
        self.check('initial server status', self.initial_status)
        self.check('model list', lambda: require(self.call('/v1/models')['data']))
        invalids = [None, '', 'stop', 1, True, {}, [''], [1], [None], ['valid', False]]
        for kind in self.cases:
            for invalid in invalids:
                self.check(f'{kind} invalid stop {invalid!r}',
                           lambda k=kind, v=invalid: self.request(k, {'stop': v}, expected=400))
            for legacy in ([], None, [0]):
                self.check(f'{kind} mutually exclusive stop_tokens={legacy!r}',
                           lambda k=kind, v=legacy: self.request(k, {'stop': [], 'stop_tokens': v}, expected=400))
            self.check(f'{kind} deterministic baseline and empty stop array', lambda k=kind: self.baseline(k))
            if kind not in self.baselines:
                continue
            prefix, full = self.baselines[kind]
            suffix = prefix[-1:]
            for stream in ((False,) if kind == 'translation' else (False, True)):
                for chunk in (1, 100):
                    self.check(f'{kind} text stop stream={stream} chunk={chunk}',
                               lambda k=kind, s=stream, c=chunk, p=prefix: self.positive(k, [p], p, s, chunk_size=c))
                self.check(f'{kind} last-budget-token stop stream={stream}',
                           lambda k=kind, s=stream, p=prefix: self.positive(k, [p], p, s, max_tokens=3, chunk_size=100))
                self.check(f'{kind} multiple candidates stream={stream}',
                           lambda k=kind, s=stream, p=prefix: self.positive(k, ['not-a-match-74ca9', p], p, s))
                self.check(f'{kind} ordered overlapping suffixes stream={stream}',
                           lambda k=kind, s=stream, p=prefix, x=suffix: self.positive(k, [p, x], self.first_suffix(k, x), s))
                self.check(f'{kind} no-match length stream={stream}',
                           lambda k=kind, s=stream: self.no_match(k, s, ['never-match-74ca9']))
                self.check(f'{kind} legacy token path stream={stream}',
                           lambda k=kind, s=stream: self.legacy(k, s))
        for stream in (False, True):
            self.check(f'batch independent rows and immediate stop stream={stream}', lambda s=stream: self.independent_batch(s))
            self.check(f'prompt text does not trigger stop stream={stream}', lambda s=stream: self.prompt_only(s))
            for literal in ('</tool_call>', '\nUser:'):
                self.check(f'literal {literal!r} stream={stream}', lambda l=literal, s=stream: self.literal_batch(l, s))
        for kind in self.cases:
            for stream in ((False,) if kind == 'translation' else (False, True)):
                self.check(f'{kind} zero token budget stream={stream}', lambda k=kind, s=stream: self.zero_budget(k, s))
                self.check(f'{kind} actual legacy token ID hit stream={stream}', lambda k=kind, s=stream: self.legacy_hit(k, s))
        for stream in (False, True):
            self.check(f'case-sensitive string stop stream={stream}', lambda s=stream: self.case_sensitive(s))
            self.check(f'only suffix matches within one token stream={stream}', lambda s=stream: self.suffix_only(s))
            self.check(f'4096 budget stops at 3 tokens and releases capacity stream={stream}', lambda s=stream: self.release_capacity(s))
            self.check(f'cached State continuation after text stop stream={stream}', lambda s=stream: self.cached_state_continuation(s))
            self.check(f'UTF-8 Chinese string stop stream={stream}', lambda s=stream: self.positive('batch', ['你好'], ' 你好', s, chunk_size=100))
        for mode in ('inherit', 'replace', 'legacy'):
            self.check(f'pause/resume stop mode={mode}', lambda m=mode: self.resume(m))
        self.check('concurrent stops across four endpoints', self.concurrent_stops)

    def zero_budget(self, kind, stream):
        result = self.request(kind, {'max_tokens': 0, 'stop': ['never-match'], 'stream': stream})
        texts, choices = self.unpack(result, stream, kind == 'translation')
        assert texts == [''] and choices[0]['finish_reason'] == 'length', (texts, choices)
        assert 'matched_stop' not in choices[0], choices

    def legacy_hit(self, kind, stream):
        first = self.unpack(self.request(kind, {'max_tokens': 1}), translation=kind == 'translation')[0][0]
        vocab = {}
        for line in (Path(__file__).resolve().parents[1] / 'assets/rwkv_vocab_v20230424.txt').read_text().splitlines():
            token_id, literal = line.split(' ', 1)
            literal, size = literal.rsplit(' ', 1)
            value = ast.literal_eval(literal)
            value = value.encode() if isinstance(value, str) else value
            vocab[value] = int(token_id)
        assert first.encode() in vocab, ('first token is not complete UTF-8', first)
        result = self.request(kind, {'stop_tokens': [vocab[first.encode()]], 'stream': stream})
        texts, choices = self.unpack(result, stream, kind == 'translation')
        assert texts == [''] and choices[0]['finish_reason'] == 'stop', (texts, choices)
        assert 'matched_stop' not in choices[0], choices

    def case_sensitive(self, stream):
        prompt = 'User: Use the calculator tool to calculate 1+1.\n\nAssistant: <tool_call>\n{"name":"calculator","arguments":{"expression":"1+1"}}\n'
        baseline = self.unpack(self.request('batch', {'contents': [prompt], 'max_tokens': 24}))[0][0]
        assert '</tool_call>' in baseline and '</TOOL_CALL>' not in baseline, baseline
        result = self.request('batch', {'contents': [prompt], 'max_tokens': 24,
                                       'stop': ['</TOOL_CALL>'], 'stream': stream})
        texts, choices = self.unpack(result, stream)
        assert texts == [baseline] and choices[0]['finish_reason'] == 'length', (texts, choices)
        assert 'matched_stop' not in choices[0], choices

    def suffix_only(self, stream):
        prompt = 'User: Repeat exactly: </tool_call>\n\nAssistant: </tool_'
        baseline = self.unpack(self.request('batch', {'contents': [prompt], 'max_tokens': 1}))[0][0]
        assert len(baseline) > 1, baseline
        substring = baseline[:-1]
        result = self.request('batch', {'contents': [prompt], 'max_tokens': 1,
                                       'stop': [substring], 'stream': stream})
        texts, choices = self.unpack(result, stream)
        assert texts == [baseline] and choices[0]['finish_reason'] == 'length', (texts, choices)
        assert 'matched_stop' not in choices[0], choices

    def release_capacity(self, stream):
        prefix = self.baselines['batch'][0]
        self.positive('batch', [prefix], prefix, stream, max_tokens=4096, metrics=True, chunk_size=100)
        # RequestRegistry finishes just after the response is sent.
        deadline = time.monotonic() + 5
        while True:
            status = self.call('/v1/server/status')
            last = status.get('last_request') or {}
            if not status['active_requests'] and last.get('max_tokens') == 4096:
                break
            assert time.monotonic() < deadline, status
            time.sleep(0.02)
        assert last['generated_tokens'] == 3 and not last['stop_requested'], last
        assert status['prefill_queue']['reserved_bsz'] == 0, status

    def cached_state_continuation(self, stream):
        path = self.cases['state'][0]
        reference = self.session('cached-reference-' + uuid.uuid4().hex[:8])
        actual = self.session('cached-stop-' + uuid.uuid4().hex[:8])
        body = self.options | self.cases['state'][1]
        first = self.unpack(self.call(path, body | {'session_id': reference, 'max_tokens': 3}))[0][0]
        continuation = {'contents': ['\n\nUser: Continue.\n\nAssistant:'], 'max_tokens': 8}
        expected = self.unpack(self.call(path, body | continuation | {'session_id': reference}))[0][0]
        assert first and expected, (first, expected)
        result = self.call(path, body | {'session_id': actual, 'max_tokens': 4096,
                                         'stop': [first], 'stream': stream, 'chunk_size': 100})
        texts, choices = self.unpack(result, stream)
        assert texts == [first] and choices[0]['matched_stop'] == first, (texts, choices)
        result = self.call(path, body | continuation | {'session_id': actual, 'max_tokens': 4096,
                                                       'stop': [expected], 'stream': stream, 'chunk_size': 100})
        texts, choices = self.unpack(result, stream)
        assert texts == [expected] and choices[0]['matched_stop'] == expected, (texts, choices, expected)
        assert choices[0]['finish_reason'] == 'stop', choices

    def first_suffix(self, kind, suffix):
        # An overlapping suffix may itself match before the longer string.
        if kind == 'translation':
            # Obtain exact decode boundaries through equivalent batch prompt.
            events = self.request('batch', {'stream': True, 'chunk_size': 1})
        else:
            events = self.request(kind, {'stream': True, 'chunk_size': 1})
        text = ''
        for event in events[:-1]:
            for c in event['choices']:
                text += c['delta'].get('content', '')
                if text.endswith(suffix):
                    return text
        raise AssertionError(('suffix not produced', kind, suffix))

    def cleanup(self):
        # Only target IDs recorded from this suite; never stop another client's request.
        status = self.call('/v1/server/status')
        for request in status['active_requests']:
            if request['id'] in self.owned_requests:
                self.call('/v1/server/stop', {'request_id': request['id']})
        for request in status['paused_requests']:
            if request['id'] in self.owned_requests:
                self.check('cleanup paused request ' + request['id'], lambda r=request: self.call(
                    '/v1/server/resume', {'request_id': r['id'], 'max_tokens': 0}))
                self.call('/state/delete', {'session_id': request['id']}, expected=(200, 404))
        super().cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--password', default=os.environ.get('PASSWORD', ''))
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--output', default='build/string-stop-http-' + time.strftime('%Y%m%d-%H%M%S'))
    args = parser.parse_args()
    suite = StringStopSuite(args)
    try:
        suite.run()
    finally:
        suite.cleanup()
        suite.save()
    report = json.loads((suite.output / 'report.json').read_text())
    print(f"Report: {suite.output / 'report.json'}; {report['passed']} passed, {report['failed']} failed; {report['requests']} requests", flush=True)
    return int(report['failed'] != 0)


if __name__ == '__main__':
    raise SystemExit(main())
