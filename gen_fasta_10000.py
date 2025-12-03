import random

def random_seq(L):
    alphabet = "ACGT"
    return "".join(random.choice(alphabet) for _ in range(L))

def main():
    n = 10000
    L = 100
    output = "dataset_10000seq.fa"

    with open(output, "w") as f:
        for i in range(n):
            seq = random_seq(L)
            f.write(f">{i}\n")
            f.write(seq[:50] + "\n")
            f.write(seq[50:] + "\n")

    print(f"Fichier '{output}' généré avec {n} séquences de longueur {L}.")

if __name__ == "__main__":
    main()