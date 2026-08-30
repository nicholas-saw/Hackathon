"""Anthropic client, prompt caching, and the token/cost meter.

Budget reality: claude-opus-5 is $5 / $25 per MTok. A node costs roughly one proposer
call, one to three coder calls and one reflector call. Uncached, ~20 iterations plus two
rehearsals runs to about $26 — over the $20 the team has. With the stable research packet
cached it is about $9. Caching is what makes the budget work, not downgrading the model,
so `assert_cache_working()` exists and the controller calls it after the second request.
"""
import json
import os
import re
import time

MODEL = 'claude-opus-5'
MAX_TOKENS = 16000

# USD per million tokens, from the current pricing table.
PRICE = {'claude-opus-5': (5.0, 25.0),
         'claude-sonnet-5': (2.0, 10.0),
         'claude-haiku-4-5': (1.0, 5.0)}
# Cache write pricing depends on the TTL: 1.25x input for the default 5-minute
# ephemeral cache, 2x for the 1-hour one. Reads are 0.1x either way.
CACHE_WRITE_MULT_5M = 1.25
CACHE_WRITE_MULT_1H = 2.00
CACHE_READ_MULT = 0.10

# A node trains for 30-95s and a confirmation run triples that, so consecutive LLM
# calls in one iteration sit minutes apart -- far enough for a 5-minute cache entry to
# expire and be rewritten. A measured 3-iteration run paid for two writes of the ~33k
# token packet. One 1-hour write at 2x costs less than two 5-minute writes at 1.25x,
# and the gap widens on a long run: a 6-hour 50-iteration run would rewrite the packet
# dozens of times at 5 minutes, versus roughly six times at an hour.
CACHE_TTL = '1h'
CACHE_WRITE_MULT = CACHE_WRITE_MULT_1H if CACHE_TTL == '1h' else CACHE_WRITE_MULT_5M


class BudgetHalt(RuntimeError):
    """Raised when the run has spent its ceiling. The controller stops cleanly."""


class LLMUnavailable(RuntimeError):
    """No SDK or no credentials. Dry runs continue; a measured run must not."""


class Meter:
    """Running token and cost accounting, per role. This IS the Feasibility submission."""

    def __init__(self, ceiling_usd=14.0):
        self.ceiling = ceiling_usd
        self.calls = []

    def add(self, role, usage, model=MODEL):
        inp = getattr(usage, 'input_tokens', 0) or 0
        out = getattr(usage, 'output_tokens', 0) or 0
        cw = getattr(usage, 'cache_creation_input_tokens', 0) or 0
        cr = getattr(usage, 'cache_read_input_tokens', 0) or 0
        pin, pout = PRICE.get(model, PRICE[MODEL])
        cost = (inp * pin + cw * pin * CACHE_WRITE_MULT + cr * pin * CACHE_READ_MULT
                + out * pout) / 1e6
        rec = {'role': role, 'model': model, 'input': inp, 'output': out,
               'cache_write': cw, 'cache_read': cr, 'usd': round(cost, 6)}
        self.calls.append(rec)
        return rec

    def totals(self):
        t = {'calls': len(self.calls), 'input': 0, 'output': 0,
             'cache_write': 0, 'cache_read': 0, 'usd': 0.0, 'by_role': {}}
        for c in self.calls:
            for k in ('input', 'output', 'cache_write', 'cache_read'):
                t[k] += c[k]
            t['usd'] += c['usd']
            r = t['by_role'].setdefault(c['role'], {'calls': 0, 'input': 0, 'output': 0,
                                                    'cache_write': 0, 'cache_read': 0,
                                                    'usd': 0.0})
            r['calls'] += 1
            r['input'] += c['input'] + c['cache_read'] + c['cache_write']
            r['output'] += c['output']
            r['cache_write'] += c['cache_write']
            r['cache_read'] += c['cache_read']
            r['usd'] = round(r['usd'] + c['usd'], 6)
        t['usd'] = round(t['usd'], 4)
        t['tokens_total'] = t['input'] + t['output'] + t['cache_read'] + t['cache_write']
        return t

    def check(self):
        spent = self.totals()['usd']
        if spent >= self.ceiling:
            raise BudgetHalt('spent $%.2f of the $%.2f ceiling' % (spent, self.ceiling))

    def cache_working(self):
        """True once any call has read from cache. False after 2+ calls means the
        prefix is being invalidated and the run will cost ~3x what was budgeted."""
        return any(c['cache_read'] > 0 for c in self.calls)


class LLM:
    """Thin wrapper: cached system packet, adaptive thinking, JSON-out with one repair."""

    def __init__(self, packet, meter=None, model=MODEL, enabled=True):
        self.packet = packet
        self.model = model
        self.meter = meter or Meter()
        self.client = None
        self.enabled = enabled
        if enabled:
            try:
                import anthropic
            except ImportError:
                raise LLMUnavailable('the `anthropic` package is not installed')
            if not (os.environ.get('ANTHROPIC_API_KEY') or os.environ.get('ANTHROPIC_AUTH_TOKEN')):
                raise LLMUnavailable(
                    'no ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN. Set one, or run the '
                    'controller with --dry-run, which uses a fixed idea list and no model.')
            # An identity-linked API key must name the workspace each request acts in,
            # or the API rejects it with 400 invalid_request_error. A classic key does
            # not need the header, so this stays absent unless the env var is set.
            headers = {}
            workspace = (os.environ.get('ANTHROPIC_WORKSPACE_ID')
                         or os.environ.get('ANTHROPIC_AWS_WORKSPACE_ID'))
            if workspace:
                headers['anthropic-workspace-id'] = workspace.strip()
            self.workspace_id = workspace
            self.client = anthropic.Anthropic(default_headers=headers or None)

    def _system(self, role_instructions):
        """Stable packet first (cached), volatile role instruction after the breakpoint.

        Caching is a prefix match, so the packet must be byte-identical on every call.
        Anything that varies per iteration goes in the user message, never in here.
        """
        return [
            {'type': 'text', 'text': self.packet,
             'cache_control': {'type': 'ephemeral', 'ttl': CACHE_TTL}},
            {'type': 'text', 'text': role_instructions},
        ]

    def ask(self, role, instructions, user, effort='high', max_tokens=MAX_TOKENS):
        if not self.enabled:
            raise LLMUnavailable('LLM disabled (dry run)')
        self.meter.check()
        t0 = time.time()
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=self._system(instructions),
            thinking={'type': 'adaptive'},
            output_config={'effort': effort},
            messages=[{'role': 'user', 'content': user}],
        )
        rec = self.meter.add(role, resp.usage, self.model)
        rec['seconds'] = round(time.time() - t0, 2)
        text = ''.join(b.text for b in resp.content if getattr(b, 'type', '') == 'text')
        return text, rec

    def ask_json(self, role, instructions, user, effort='high', tries=2):
        """Ask for JSON and parse it. One repair attempt, then give up honestly."""
        last = None
        for attempt in range(tries):
            msg = user if attempt == 0 else (
                user + '\n\nYour previous reply did not parse as JSON (%s). '
                'Reply with the JSON object only — no prose, no code fence.' % last)
            text, rec = self.ask(role, instructions, msg, effort=effort)
            try:
                return extract_json(text), rec
            except Exception as exc:
                last = str(exc)[:200]
        raise ValueError('%s produced unparseable JSON twice: %s' % (role, last))


def extract_json(text):
    """Pull the first JSON object out of a reply, fence or no fence."""
    t = text.strip()
    fence = re.search(r'```(?:json)?\s*(.+?)```', t, re.S)
    if fence:
        t = fence.group(1).strip()
    start = t.find('{')
    if start == -1:
        raise ValueError('no JSON object in reply')
    depth, in_str, esc = 0, False, False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return json.loads(t[start:i + 1])
    raise ValueError('unbalanced JSON braces')
