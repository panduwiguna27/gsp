#!/usr/bin/env python3
import pandas as pd
import os
from sentence_transformers import SentenceTransformer, util

class BotLieDetector:
    def __init__(self, csv_path, policies_folder):
        self.csv_path = csv_path
        self.policies_folder = policies_folder
        self.df = pd.read_csv(csv_path)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.policies = self._load_policies()
        self.policy_embeddings = self._encode_policies()

    def _load_policies(self):
        policies = []
        if not os.path.exists(self.policies_folder):
            print('Warning: policies folder not found')
            return policies
        for filename in os.listdir(self.policies_folder):
            if filename.endswith('.txt'):
                filepath = os.path.join(self.policies_folder, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        policies.append({'filename': filename, 'content': content})
        return policies

    def _encode_policies(self):
        if not self.policies:
            return []
        contents = [p['content'] for p in self.policies]
        embeddings = self.model.encode(contents, convert_to_tensor=True)
        return list(zip(self.policies, embeddings))

    def _check_answer(self, answer):
        if not answer or not isinstance(answer, str):
            return 0.0
        if not self.policies:
            return None
        answer_emb = self.model.encode(answer, convert_to_tensor=True)
        similarities = []
        for _, emb in self.policy_embeddings:
            sim = util.cos_sim(answer_emb, emb).item()
            similarities.append(sim)
        return max(similarities) if similarities else 0.0

    def analyze(self, threshold=0.3):
        results = []
        total = len(self.df)
        print(f'Analyzing {total} turns...')
        for _, row in self.df.iterrows():
            q = row['question'] or ''
            a = row['answer'] or ''
            score = self._check_answer(a)
            if score is None:
                flag = 'Unknown'
                desc = 'No policies'
            elif score >= threshold:
                flag = 'Truthful'
                desc = f'High: {score:.2f}'
            else:
                flag = 'Lying Risk'
                desc = f'Low: {score:.2f}'
            results.append({
                'id': row['id'], 'question': q[:60], 'answer': a[:80],
                'score': round(score, 4) if score is not None else None,
                'flag': flag, 'note': desc
            })
        res_df = pd.DataFrame(results)
        total_cnt = len(res_df)
        lying_count = len(res_df[res_df['flag'] == 'Lying Risk'])
        truthful_count = len(res_df[res_df['flag'] == 'Truthful'])
        unknown_count = len(res_df[res_df['flag'] == 'Unknown'])
        print('='*60)
        print('LIE ANALYSIS SUMMARY')
        print('='*60)
        print(f'Total: {total_cnt}')
        print(f'Truthful: {truthful_count} ({truthful_count/total_cnt*100:.1f}%)')
        print(f'Lying Risk: {lying_count} ({lying_count/total_cnt*100:.1f}%)')
        print(f'Unknown: {unknown_count} ({unknown_count/total_cnt*100:.1f}%)')
        print()
        lying = res_df[res_df['flag'] == 'Lying Risk']
        if len(lying) > 0:
            print('Lying Risk Details:')
            for _, r in lying.iterrows():
                print(f"  - ID {r['id']}: {r['question'][:50]}... | {r['answer'][:60]}... | Score: {r['score']}")
        print('='*60)
        return res_df

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'conversations.csv')
    policies_folder = os.path.join(script_dir, 'policies')
    detector = BotLieDetector(csv_path, policies_folder)
    results = detector.analyze(threshold=0.3)
    results.to_csv(os.path.join(script_dir, 'lie_analysis_results.csv'), index=False)
    print('Saved results to lie_analysis_results.csv')

if __name__ == '__main__':
    main()