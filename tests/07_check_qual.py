# quick_check_qual.py
from dotenv import load_dotenv
load_dotenv()
from app.tools.qualitative_analysis_tool import QualitativeAnalysisTool

# Use a toy embedder
class TinyEmbedder:
    def encode(self, texts, show_progress_bar=False):
        return [[float(len(t)) for _ in range(8)] for t in texts]

tool = QualitativeAnalysisTool(embedder=TinyEmbedder())
transcripts = [{"name": "t1", "local_path": "tests/data/sample_transcript_q1.txt"}]
ok = tool.index_transcripts(transcripts)
print('index ok:', ok)
print('retrieve demand:', tool.retrieve('demand', top_k=3))
print('analysis:', tool.analyze(transcripts))