from .tree import is_tree_eq

def eval(pred_file: str, gold_file: str):
    correct, total = 0, 0
    for (pred, gold) in zip(open(pred_file), open(gold_file)):
        if is_tree_eq(pred, gold, not_layout=True):
            correct += 1
        total += 1

    print("pred_file: ", pred_file)
    print("gold_file: ", gold_file)
    print("exact match:", correct, "/", total, "=", correct * 1. / total)

if __name__ == '__main__':
    import sys
    eval(sys.argv[1], sys.argv[2])

