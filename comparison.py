import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

with open("/content/drive/MyDrive/TikTok-Study/llava_next_analysis.json") as f:
    results1 = {r["video_id"]: r for r in json.load(f)}

with open("/content/drive/MyDrive/TikTok-Study/llava_next_analysis1.json") as f:
    results2 = {r["video_id"]: r for r in json.load(f)}

emb_model = SentenceTransformer("all-MiniLM-L6-v2")

TEXT_FIELDS  = ["content_summary", "emotional_tone", "persuasion_techniques",
                "target_audience", "credibility_assessment", "misinformation_risk",
                "behavioral_impact", "key_message"]
LABEL_FIELDS = ["content_category", "risk_level"]

common_ids = set(results1.keys()) & set(results2.keys())
print(f"Comparing {len(common_ids)} videos present in both outputs\n")

field_scores  = {f: [] for f in TEXT_FIELDS}
label_matches = {f: [] for f in LABEL_FIELDS}

for vid_id in common_ids:
    a1 = results1[vid_id]["llava_analysis"]
    a2 = results2[vid_id]["llava_analysis"]

    for field in TEXT_FIELDS:
        t1, t2 = a1.get(field, ""), a2.get(field, "")
        if t1 and t2:
            e1 = emb_model.encode([t1])
            e2 = emb_model.encode([t2])
            score = cosine_similarity(e1, e2)[0][0]
            field_scores[field].append(score)

    for field in LABEL_FIELDS:
        t1 = a1.get(field, "").lower()
        t2 = a2.get(field, "").lower()
        match = t1.split()[0] == t2.split()[0] if t1 and t2 else False
        label_matches[field].append(match)

print("📊 EMBEDDING SIMILARITY (higher = more agreement)")
print("-" * 50)
all_scores = []
for field, scores in field_scores.items():
    if scores:
        avg = sum(scores) / len(scores)
        all_scores.append(avg)
        print(f"  {field:<30} {avg:.3f}")

print(f"\n  {'OVERALL AVERAGE':<30} {sum(all_scores)/len(all_scores):.3f}")

print("\n📊 LABEL AGREEMENT (exact match)")
print("-" * 50)
for field, matches in label_matches.items():
    if matches:
        rate = sum(matches) / len(matches)
        print(f"  {field:<30} {rate:.1%}  ({sum(matches)}/{len(matches)} videos)")