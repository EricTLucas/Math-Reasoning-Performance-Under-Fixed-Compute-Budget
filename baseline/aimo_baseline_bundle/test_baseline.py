"""CPU-only tests. Run: python test_baseline.py"""
import argparse
import csv
import json
from pathlib import Path
import tempfile
import unittest
import aimo_baseline as b

class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.data = self.base / 'input.jsonl'
        self.data.write_text('\n'.join(json.dumps(r) for r in [
            {'id': 1, 'problem': 'One?', 'answer': '70'},
            {'id': 2, 'problem': 'Two?', 'answer': '0'}]))
        model = self.base / 'model'; model.mkdir()
        (model / 'config.json').write_text('{}')
        (model / 'DOWNLOAD_REVISION.txt').write_text('test-revision')
        self.root = self.base / 'experiment'
        b.prepare(argparse.Namespace(data=str(self.data), model=str(model), experiment=str(self.root), chunk_size=1, context=4096, seconds_per_attempt=300))
        self.cfg, self.rows = b.load_experiment(self.root)
    def save(self, i, a, answer, status='answered'):
        r = {'row_index': i, 'problem_id': self.rows[i]['id'], 'attempt': a,
             'reference_answer': self.rows[i]['answer'], 'predicted_answer': answer,
             'correct': answer == self.rows[i]['answer'], 'status': status,
             'elapsed_seconds': 10*a, 'entropy': .5, 'seed': a}
        b.atomic_json(b.record_path(self.root,i,a),r)
    def test_summary_and_missing(self):
        self.save(0,1,70); self.save(0,2,68); self.save(0,3,None,'timeout')
        self.save(1,1,0)
        b.export(argparse.Namespace(experiment=str(self.root)))
        out=self.root/'exports'
        with (out/'baseline_summary.csv').open(encoding='utf-8-sig') as f:
            rows=list(csv.DictReader(f))
        self.assertEqual(rows[0]['correct_attempts'],'1')
        self.assertEqual(rows[0]['wrong_answers'],'[68]')
        self.assertEqual(rows[0]['total_attempt_seconds'],'60')
        self.assertEqual(rows[0]['failed_attempts'],'1')
        self.assertEqual(rows[1]['complete'],'False')
        self.assertEqual(rows[1]['attempt_1_answer'],'0')
        self.assertEqual(rows[1]['attempt_2_status'],'missing')
        self.assertEqual(rows[1]['voted_answer'],'')
        self.assertEqual(json.loads((out/'completion.json').read_text())['missing_tasks'],[2])
    def test_resume_and_partial_write(self):
        self.save(0,1,70)
        p=b.record_path(self.root,0,2); p.with_name(p.name+'.tmp.999').write_text('{')
        self.assertIsNotNone(b.read_record(self.root,0,1,self.rows[0]))
        self.assertIsNone(b.read_record(self.root,0,2,self.rows[0]))
        self.save(0,2,None,'timeout')
        self.assertEqual(b.read_record(self.root,0,2,self.rows[0])['status'],'timeout')
    def test_dataset_guard_and_duplicates(self):
        p=self.root/'dataset.jsonl'; p.write_text(p.read_text()+'\n')
        with self.assertRaises(ValueError): b.load_experiment(self.root)
        self.data.write_text('{"id":1,"problem":"a","answer":0}\n'*2)
        with self.assertRaises(ValueError): b.rows_from(self.data)
    def test_complete_count(self):
        for i in range(2):
            for a in range(1,4): self.save(i,a,self.rows[i]['answer'])
        b.export(argparse.Namespace(experiment=str(self.root)))
        report=json.loads((self.root/'exports/completion.json').read_text())
        self.assertEqual(report['saved_attempts'],6)
        self.assertEqual(report['complete_problems'],2)
        self.assertEqual(report['missing_tasks'],[])

if __name__=='__main__': unittest.main()
