# =========================================
# FINAL CLASSROOM SOCIAL NETWORK PROJECT ✅
# =========================================

import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import re

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------
print("📥 Loading dataset...")
nodes = pd.read_csv("data/nodes.csv")
edges = pd.read_csv("data/edges.csv")
print(f"✅ Loaded: {nodes.shape[0]} students, {edges.shape[0]} connections")

# --------------------------------------------------
# 2. CLEAN + PARSE SKILLS
# --------------------------------------------------
def parse_skills(s):
    if pd.isna(s):
        return set()
    parts = re.split(r"[;,|]", str(s))
    return {p.strip().lower() for p in parts if p.strip()}

nodes["skills_set"] = nodes["skills"].apply(parse_skills)
nodes["n_skills"] = nodes["skills_set"].apply(len)

# Unique string IDs
nodes["id"] = nodes["id"].astype(str)
edges["source"] = edges["source"].astype(str)
edges["target"] = edges["target"].astype(str)

# --------------------------------------------------
# 3. CREATE TEXT FIELD + COSINE SIMILARITY
# --------------------------------------------------
print("📊 Computing TF–IDF cosine similarity...")

nodes["text_for_sim"] = (
    nodes["skills"].fillna("") + " " + nodes["target"].fillna("")
)

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(nodes["text_for_sim"])
cos_sim_matrix = cosine_similarity(tfidf_matrix)

# --------------------------------------------------
# 4. BUILD GRAPH
# --------------------------------------------------
print("🔗 Building network graph...")

G = nx.Graph()
id_to_index = {id_: idx for idx, id_ in enumerate(nodes["id"])}

# Add nodes
for _, r in nodes.iterrows():
    G.add_node(
        r["id"],
        name=r["name"],
        reg_no=r["reg_no"],
        section=r["section"],
        skills_set=r["skills_set"],
        skills=r["skills"],
        target=r["target"],
        n_skills=r["n_skills"]
    )

# Add edges with weights = strength + cosine_similarity
for _, r in edges.iterrows():
    u, v = r["source"], r["target"]
    if u not in G or v not in G:
        continue

    idx_u = id_to_index[u]
    idx_v = id_to_index[v]
    cos_sim = cos_sim_matrix[idx_u][idx_v]

    strength = float(r["strength"]) if not pd.isna(r["strength"]) else 5.0
    final_weight = 0.5 * strength + 0.5 * (cos_sim * 10)

    G.add_edge(
        u, v,
        weight=final_weight,
        raw_strength=strength,
        cos_sim=cos_sim,
        relation=r["connection_type"]
    )

print(f"✅ Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# --------------------------------------------------
# 5. SKILL SELECTION
# --------------------------------------------------
print("\n🎯 Example skills:", nodes["skills"].dropna().sample(5).tolist())
skill_input = input("\nEnter a skill to simulate influence: ").lower().strip()

print(f"\n🔎 Searching for students with skill: {skill_input}")

# Students who HAVE the skill
skill_students = nodes[nodes["skills_set"].apply(lambda s: skill_input in s)]["id"].tolist()

if skill_students:
    print(f"✅ Found {len(skill_students)} students with this skill.")
    use_skill_seeds = True
else:
    print("❌ No student has this skill, using top-degree nodes.")
    use_skill_seeds = False

# Pick seeds
if use_skill_seeds:
    skill_df = nodes[nodes["id"].isin(skill_students)].copy()
    skill_df["degree"] = skill_df["id"].apply(lambda x: G.degree(x))
    skill_df.sort_values("degree", ascending=False, inplace=True)
    seed_nodes = skill_df["id"].head(5).tolist()
else:
    deg = dict(G.degree())
    seed_nodes = sorted(deg, key=deg.get, reverse=True)[:5]

print("🔥 Seed nodes:", seed_nodes)

# --------------------------------------------------
# 6. SMART INFLUENCE CASCADE
# --------------------------------------------------
def independent_cascade_smart(G, seeds, skill, base_p=0.08, steps=15):
    influenced = set(seeds)
    newly = set(seeds)
    timeline = [len(influenced)]
    timesteps = [set(seeds)]
    skill = skill.lower().strip()

    for _ in range(steps):
        next_wave = set()

        for u in newly:
            for v in G.neighbors(u):
                if v in influenced:
                    continue

                w = G[u][v]["weight"]
                # 1. Skill-based multiplier
                has_skill = skill in G.nodes[v]["skills_set"]
                skill_mult = 2.0 if has_skill else 0.7

                # 2. Role-based multiplier (target similarity)
                u_idx = id_to_index[u]
                v_idx = id_to_index[v]
                role_sim = cos_sim_matrix[u_idx][v_idx]  # 0–1

                role_mult = 1 + (role_sim * 1.2)   # up to +120% boost

                # 3. If the student's target headline explicitly mentions the skill
                headline = str(G.nodes[v]["target"]).lower()
                headline_bonus = 1.35 if skill in headline else 1.0

                # Final probability
                prob = base_p * (w / 10.0) * skill_mult * role_mult * headline_bonus
                prob = min(prob, 0.55)   # hard cap


                if np.random.rand() < prob:
                    next_wave.add(v)

        if not next_wave:
            break

        influenced |= next_wave
        newly = next_wave
        timeline.append(len(influenced))
        timesteps.append(set(next_wave))

    return influenced, timeline, timesteps

print("\n🚀 Simulating influence...")
influenced, timeline, timesteps = independent_cascade_smart(G, seed_nodes, skill_input)

print(f"✅ Influence reached {len(influenced)} students.")

# --------------------------------------------------
# 7. SYNCHRONIZE VISUALS = FORCE INCLUDE SKILL STUDENTS
# --------------------------------------------------
viz_nodes = list(set(influenced) | set(skill_students))
viz_nodes = viz_nodes[:400]  # safe limit

subG = G.subgraph(viz_nodes).copy()

# --------------------------------------------------
# 8. PYVIS NETWORK VISUALIZATION
# --------------------------------------------------
print("\n🎨 Building interactive network...")

nt = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black", cdn_resources="remote")
nt.barnes_hut(gravity=-25000, central_gravity=0.3, spring_length=180, spring_strength=0.02, damping=0.28)

# Section colors
section_colors = {
    "A": "#1f77b4",   # blue
    "B": "#2ecc71",   # green
    "C": "#f39c12"    # orange
}

for node, data in subG.nodes(data=True):
    size = 12 + (subG.degree(node) * 1.3)

    section = str(data.get("section", "")).upper()
    color = section_colors.get(section, "#7f8c8d")  # default grey if missing

    label = f"{data['name']} ({node})"
    tooltip = (
        f"<b>{data['name']}</b><br>"
        f"Reg No: {data['reg_no']}<br>"
        f"Section: {section}<br>"
        f"Skills: {data['skills']}<br>"
        f"Target: {data['target']}"
    )

    nt.add_node(
        node,
        label=label,
        size=size,
        color=color,
        title=tooltip
    )

for u, v, ed in subG.edges(data=True):
    nt.add_edge(u, v, value=ed["weight"], title=f"{ed.get('relation', '')} (w={ed['weight']:.2f})")

nt.write_html("influence_network.html")
print("✅ Saved → influence_network.html")

# --------------------------------------------------
# 9. DIFFUSION CURVE (Synced with viz_nodes)
# --------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(
    range(len(timeline)),
    [min(x, len(viz_nodes)) for x in timeline],
    marker="o",
    color="#2ecc71"
)
plt.title(f"Diffusion Curve — Skill: {skill_input}")
plt.xlabel("Time Step")
plt.ylabel("Influenced Students")
plt.grid(True)
plt.tight_layout()
plt.savefig("diffusion_curve.png")
plt.show()
print("✅ Saved → diffusion_curve.png")

# --------------------------------------------------
# 10. PLOTLY ANIMATION (SYNCHRONIZED)
# --------------------------------------------------
print("\n🎥 Creating diffusion animation...")

import plotly.graph_objects as go

nodes_list = viz_nodes
pos = nx.spring_layout(subG, seed=42)

x = [pos[n][0] if n in pos else 0 for n in nodes_list]
y = [pos[n][1] if n in pos else 0 for n in nodes_list]
node_names = [subG.nodes[n].get("name", n) for n in nodes_list]

frames = []
cumulative = set()

for step_idx, step_nodes in enumerate(timesteps):
    cumulative |= step_nodes

    colors = ["#2ecc71" if n in cumulative else "#3498db" for n in nodes_list]

    frames.append(go.Frame(
        data=[go.Scatter(
            x=x, y=y, mode="markers+text",
            marker=dict(size=[10 + subG.degree(n) for n in nodes_list], color=colors),
            text=[node_names[i] if nodes_list[i] in cumulative else "" for i in range(len(nodes_list))],
            textposition="top center",
            hovertext=[f"{node_names[i]} ({nodes_list[i]})" for i in range(len(nodes_list))],
            hoverinfo="text"
        )],
        name=f"step{step_idx}"
    ))

initial_colors = ["#2ecc71" if n in timesteps[0] else "#3498db" for n in nodes_list]

fig = go.Figure(
    data=[go.Scatter(
        x=x, y=y, mode="markers+text",
        marker=dict(size=[10 + subG.degree(n) for n in nodes_list], color=initial_colors),
        text=["" for _ in nodes_list],
        hovertext=[f"{node_names[i]} ({nodes_list[i]})" for i in range(len(nodes_list))],
        hoverinfo="text"
    )],
    layout=go.Layout(
        title="Influence Spread Animation",
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(showgrid=False, visible=False),
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(
                label="Play",
                method="animate",
                args=[None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]
            )]
        )]
    ),
    frames=frames
)

fig.write_html("diffusion_animation.html")
print("✅ Saved → diffusion_animation.html")

# --------------------------------------------------
# 11. SUMMARY
# --------------------------------------------------
print("\n===================")
print("✅ FINAL SUMMARY")
print("===================")
print("Skill:", skill_input)
print("Seeds:", seed_nodes)
print("Total influenced:", len(influenced))
print("Nodes visualized:", len(viz_nodes))
print("Timeline:", timeline)

print("\nFiles generated:")
print("✅ influence_network.html")
print("✅ diffusion_curve.png")
print("✅ diffusion_animation.html")
