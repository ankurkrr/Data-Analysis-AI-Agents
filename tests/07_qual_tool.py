# tests/07_qual_tool.py
"""Run a quick smoke test for QualitativeAnalysisTool without downloading large models.

This test injects a lightweight dummy embedder to avoid network downloads and heavy CPU work
so it produces immediate output in CI/dev environments.
"""
import tests.bootstrap_path
import os
import sys
import traceback

# Early debug print to ensure the script is executed
print('TEST START: 07_qual_tool running')
sys.stdout.flush()

# Ensure importing 'sentence_transformers' is fast and cannot block: inject a dummy module
print('Injecting dummy sentence_transformers module to avoid import delays')
sys.stdout.flush()
import types
class _DummyEmbedder:
    def __init__(self, *args, **kwargs):
        pass
    def encode(self, texts, show_progress_bar=False):
        # return simple list embeddings (no numpy needed)
        emb_dim = 8
        out = [[float(len(t) if isinstance(t, str) else 1) for _ in range(emb_dim)] for t in texts]
        return out

dummy_mod = types.ModuleType('sentence_transformers')
dummy_mod.SentenceTransformer = _DummyEmbedder
sys.modules['sentence_transformers'] = dummy_mod
print('Injected dummy sentence_transformers into sys.modules')
sys.stdout.flush()

import importlib

# Use a lightweight dummy implementation to avoid heavy dependencies (sentence-transformers/faiss)
try:
    mod = importlib.import_module('app.tools.qualitative_analysis_tool')
except Exception as e:
    print('ERROR importing qualitative_analysis_tool:', e)
    traceback.print_exc()
    # write to a small log file to help debugging in editors/CI
    try:
        with open('tests/07_qual_tool_error.log', 'w', encoding='utf-8') as _f:
            _f.write('Import error:\n')
            _f.write(str(e) + '\n')
            _f.write(traceback.format_exc())
    except Exception:
        pass
    raise

class DummyQualitativeAnalysisTool:
    def __init__(self, *args, **kwargs):
        self.chunks = []
        self.index = True

    def index_transcripts(self, transcripts):
        self.chunks = []
        for t in transcripts:
            try:
                with open(t.get('local_path'), 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
            except Exception:
                text = ''
            if text:
                self.chunks.append({
                    'meta': {'chunk_id': f"{t.get('name')}_0", 'source': t.get('name')},
                    'text': text
                })
        return len(self.chunks) > 0

    def retrieve(self, query, top_k=5):
        # simple keyword match
        q = query.lower()
        results = []
        for c in self.chunks:
            if q.split()[0] in c['text'].lower() or any(word in c['text'].lower() for word in q.split(',')):
                results.append({
                    'chunk_id': c['meta']['chunk_id'],
                    'source': c['meta']['source'],
                    'text': c['text'][:600],
                    'score': 0.0
                })
        return results[:top_k]

    def analyze(self, transcripts):
        themes = []
        for theme in ['demand', 'attrition', 'guidance']:
            res = self.retrieve(theme)
            if res:
                themes.append({'theme': theme, 'count': len(res), 'examples': res[:3]})
        sentiment = {'score': 0.5, 'summary': 'neutral'}
        forward = self.retrieve('guidance')
        return {
            'tool': 'QualitativeAnalysisTool',
            'themes': themes,
            'management_sentiment': sentiment,
            'forward_guidance': forward,
            'risks': []
        }

# We'll use the real QualitativeAnalysisTool but inject a lightweight embedder instance

# create a small transcript file under tests/data/ if not present
os.makedirs('tests/data', exist_ok=True)
sample_path = 'tests/data/sample_transcript_q1.txt'
if not os.path.exists(sample_path):
    with open(sample_path, 'w', encoding='utf-8') as f:
        f.write('This is a short sample transcript about demand and growth. The company expects growth next quarter.')

transcripts = [{"name":"t1","local_path": sample_path}]

print('Instantiating QualitativeAnalysisTool (with injected dummy embedder)...')
sys.stdout.flush()
tool = mod.QualitativeAnalysisTool(embedder=_DummyEmbedder())
print('Indexing transcripts...')
sys.stdout.flush()
ok = tool.index_transcripts(transcripts)
print('index built:', ok)
if ok:
    res = tool.retrieve('demand', top_k=3)
    print('retrieve:', res)
    an = tool.analyze(transcripts)
    import json
    print('analysis:', json.dumps(an, indent=2))
    sys.stdout.flush()

print('TEST END: 07_qual_tool completed')
sys.stdout.flush()
