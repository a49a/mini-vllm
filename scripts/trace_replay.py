#!/usr/bin/env python3
"""Render teaching JSONL as a self-contained, offline request timeline."""
import argparse
import json
from pathlib import Path


def render(events):
    if any(event.get('schema_version') != 1 for event in events):
        raise ValueError('unsupported trace schema')
    payload = json.dumps(events, ensure_ascii=True).replace('<', '\\u003c')
    return r'''<!doctype html><html lang="en"><meta charset="utf-8"><title>mini-vllm trace replay</title>
<style>body{font:16px system-ui;margin:32px;background:#111827;color:#e5e7eb}button,input{margin:8px}input{width:65%}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:10px;border-bottom:1px solid #374151}.bar{height:12px;background:#34d399;border-radius:4px}pre{white-space:pre-wrap}small{color:#9ca3af}</style>
<h1>Request timeline / 请求时间线</h1><p>Scrub or play to inspect scheduler state. 同一 step 表示同一批。页数为每请求页表，非全局唯一物理页数。</p>
<button id="play">Play / 播放</button><input type="range" id="cursor" min="0" value="0"><span id="time"></span>
<p id="warning"></p><table><thead><tr><th>Request</th><th>State / 状态</th><th>Step</th><th>Position / 位置</th><th>Pages / 页</th><th>Shared prefix</th><th>Progress</th></tr></thead><tbody id="rows"></tbody></table><pre id="detail"></pre>
<script>const events=PAYLOAD; const slider=document.getElementById('cursor');slider.max=Math.max(0,events.length-1);
const rows=document.getElementById('rows'), detail=document.getElementById('detail');let timer=null;
function draw(){const i=Number(slider.value), states=new Map();for(const e of events.slice(0,i+1)){if(typeof e.request_id!=='string')continue; const s=states.get(e.request_id)||{}; Object.assign(s,e);states.set(e.request_id,s)}
rows.replaceChildren();let maximum=1;for(const s of states.values())maximum=Math.max(maximum,(s.position||0)+(s.tokens||0));
for(const [id,s] of states){const row=document.createElement('tr');for(const v of [id,['scheduled','computed'].includes(s.event)?s.event+' '+s.phase:s.event,s.step??'',s.position??'',s.event==='retired'?0:(s.pages_after??''),s.shared_prefix_tokens??0]){const cell=document.createElement('td');cell.textContent=String(v);row.append(cell)}const cell=document.createElement('td'),bar=document.createElement('div');bar.className='bar';bar.style.width=100*((s.position||0)+(s.tokens||0))/maximum+'%';cell.append(bar);row.append(cell);rows.append(row)}
const event=events[i];document.getElementById('time').textContent=event?((event.elapsed_us||0)/1000).toFixed(2)+' ms':'';detail.textContent=JSON.stringify(event||{},null,2);document.getElementById('warning').textContent=events.some(e=>e.event==='truncated')?'Trace truncated at event limit / 追踪达到条数上限':'';}
slider.oninput=draw;document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;return}if(Number(slider.value)>=Number(slider.max))slider.value=0;timer=setInterval(()=>{slider.value=Math.min(Number(slider.max),Number(slider.value)+1);draw();if(slider.value===slider.max){clearInterval(timer);timer=null}},200)};draw();</script></html>'''.replace('PAYLOAD', payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace')
    parser.add_argument('--output', default='artifacts/trace.html')
    args = parser.parse_args()
    events = [json.loads(line) for line in Path(args.trace).read_text().splitlines() if line.strip()]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(events))
    print(output.resolve())


if __name__ == '__main__':
    main()
