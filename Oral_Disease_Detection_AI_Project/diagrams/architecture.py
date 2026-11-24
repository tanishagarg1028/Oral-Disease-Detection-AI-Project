import graphviz

dot = graphviz.Digraph("architecture", format="png")
dot.attr(rankdir="TB", bgcolor="white", fontname="Helvetica", nodesep="0.3", ranksep="0.35")
dot.attr("node", shape="box", style="rounded,filled", fontname="Helvetica",
         fontsize="12", margin="0.2,0.1")

layers = [
    ("input", "Input Image\n224 x 224 x 3", "#EEEEEE"),
    ("base", "Pretrained Backbone\n(EfficientNetB0 / ResNet50)\nfrozen -> fine-tuned", "#DCEBFA"),
    ("gap", "Global Average Pooling", "#DCEBFA"),
    ("drop1", "Dropout (0.3)", "#F0F0F0"),
    ("dense", "Dense (128, ReLU)", "#D6F0E4"),
    ("drop2", "Dropout (0.2)", "#F0F0F0"),
    ("out", "Dense (Softmax)\nhealthy / cavity /\ngum disease / oral lesion", "#F5DCE3"),
]

for key, label, color in layers:
    dot.node(key, label, fillcolor=color, color="#888888")

keys = [l[0] for l in layers]
for a, b in zip(keys, keys[1:]):
    dot.edge(a, b, color="#666666")

dot.render("/home/claude/oral_disease_ai/outputs/architecture_diagram", cleanup=True)
print("saved")
