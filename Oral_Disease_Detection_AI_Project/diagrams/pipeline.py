import graphviz

dot = graphviz.Digraph("pipeline", format="png")
dot.attr(rankdir="LR", bgcolor="white", fontname="Helvetica", nodesep="0.4", ranksep="0.6")
dot.attr("node", shape="box", style="rounded,filled", fontname="Helvetica",
         fontsize="12", margin="0.2,0.12")

stages = [
    ("data", "Data Collection\n(X-rays / intraoral photos)", "#DCEBFA"),
    ("prep", "Preprocessing\n(resize, augment,\nnormalize)", "#DCEBFA"),
    ("model", "Transfer Learning Model\n(EfficientNet / ResNet CNN)", "#D6F0E4"),
    ("classify", "Classification\n(healthy / cavity /\ngum disease / lesion)", "#D6F0E4"),
    ("gradcam", "Grad-CAM\nExplainability Heatmap", "#FCE8D6"),
    ("ui", "Interface\n(Streamlit / Flask App)", "#F5DCE3"),
]

for key, label, color in stages:
    dot.node(key, label, fillcolor=color, color="#888888")

for a, b in zip([s[0] for s in stages], [s[0] for s in stages][1:]):
    dot.edge(a, b, color="#666666")

dot.render("/home/claude/oral_disease_ai/outputs/pipeline_diagram", cleanup=True)
print("saved")
