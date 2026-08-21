import json, urllib.request, time

# Load benchmark queries
with open(r'c:\Users\shana\OneDrive\Desktop\goa_hackathon\data\raw\benchmark_queries.json', 'r', encoding='utf-8') as f:
    all_queries = json.load(f)

# Take first 50
queries = all_queries[:50]

results = []
pass_count = 0
fail_count = 0
emb_times = []
gen_times = []
tot_times = []

for i, q_obj in enumerate(queries):
    eng = q_obj.get('eng_query', q_obj.get('query', ''))
    data = json.dumps({'query': eng, 'strategy': 'fixed', 'top_k': 2, 'target_lang': 'Auto-detect'}).encode('utf-8')
    req = urllib.request.Request('http://127.0.0.1:8000/api/query/text', data=data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            res = json.loads(resp.read().decode('utf-8'))
            emb = res.get('stage_latencies', {}).get('embedding', 0)
            gen = res.get('stage_latencies', {}).get('generation', 0)
            tot = res.get('pipeline_latency_ms', 0)
            emb_times.append(emb)
            gen_times.append(gen)
            tot_times.append(tot)
            status = 'PASS' if tot < 200 else 'FAIL'
            if tot < 200:
                pass_count += 1
            else:
                fail_count += 1
            label = eng[:50]
            print(f'[{i+1:2d}] [{status}] Emb:{emb:7.1f}ms Gen:{gen:7.1f}ms Tot:{tot:7.1f}ms | {label}')
    except Exception as e:
        fail_count += 1
        err_msg = str(e)[:60]
        label = eng[:50]
        print(f'[{i+1:2d}] [ERR ] {err_msg} | {label}')

print()
print('=' * 80)
print(f'RESULTS: {pass_count} PASS / {fail_count} FAIL out of {len(queries)} queries')
pct = pass_count / len(queries) * 100
print(f'Pass Rate: {pct:.1f}%')
print()
if emb_times:
    eavg = sum(emb_times) / len(emb_times)
    emin = min(emb_times)
    emax = max(emb_times)
    ep95 = sorted(emb_times)[int(len(emb_times) * 0.95)]
    print(f'Embedding:  avg={eavg:.1f}ms  min={emin:.1f}ms  max={emax:.1f}ms  p95={ep95:.1f}ms')
if gen_times:
    gavg = sum(gen_times) / len(gen_times)
    gmin = min(gen_times)
    gmax = max(gen_times)
    gp95 = sorted(gen_times)[int(len(gen_times) * 0.95)]
    print(f'Generation: avg={gavg:.1f}ms  min={gmin:.1f}ms  max={gmax:.1f}ms  p95={gp95:.1f}ms')
if tot_times:
    tavg = sum(tot_times) / len(tot_times)
    tmin = min(tot_times)
    tmax = max(tot_times)
    tp95 = sorted(tot_times)[int(len(tot_times) * 0.95)]
    print(f'Total:      avg={tavg:.1f}ms  min={tmin:.1f}ms  max={tmax:.1f}ms  p95={tp95:.1f}ms')
print()
# Show any failures
if fail_count > 0:
    print('Failing queries (>200ms):')
    for idx, t in enumerate(tot_times):
        if t >= 200:
            label = queries[idx].get('eng_query', '')[:60]
            print(f'  [{idx+1}] {t:.1f}ms | {label}')
