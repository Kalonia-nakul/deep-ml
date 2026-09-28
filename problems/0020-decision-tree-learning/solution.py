import math
from collections import Counter

def calculate_entropy(labels: list) -> float:
    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels)
    # p * log2(1/p) equals -p * log2(p) but avoids printing -0.0 for pure nodes
    return sum((c / n) * math.log2(n / c) for c in counts.values())

def calculate_information_gain(examples: list[dict], attr: str, target_attr: str) -> float:
    total = len(examples)
    parent_entropy = calculate_entropy([e[target_attr] for e in examples])

    remainder = 0.0
    for v in set(e[attr] for e in examples):
        subset_labels = [e[target_attr] for e in examples if e[attr] == v]
        remainder += (len(subset_labels) / total) * calculate_entropy(subset_labels)

    return parent_entropy - remainder

def majority_class(examples: list[dict], target_attr: str) -> str:
    counts = Counter(e[target_attr] for e in examples)
    top = max(counts.values())
    # among the classes tied for the highest count, pick alphabetically first
    return min(label for label, c in counts.items() if c == top)

def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
    labels = [e[target_attr] for e in examples]

    # Base case 1: all examples share one class -> leaf
    if len(set(labels)) == 1:
        return labels[0]

    # Base case 2: no attributes left to split on -> majority leaf
    if not attributes:
        return majority_class(examples, target_attr)

    # Pick the attribute with the highest information gain
    best = max(attributes, key=lambda a: calculate_information_gain(examples, a, target_attr))
    remaining = [a for a in attributes if a != best]

    tree = {best: {}}
    for v in sorted(set(e[best] for e in examples)):
        subset = [e for e in examples if e[best] == v]
        tree[best][v] = learn_decision_tree(subset, remaining, target_attr)
    return tree