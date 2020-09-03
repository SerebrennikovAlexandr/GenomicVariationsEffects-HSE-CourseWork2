import pandas as pd
from random import sample, seed


# ----- Settings part -----


PATH_TO_HG = r"../../hg_assemblies/hg19/hg19_"
PATH_TO_TISSUE_FILE = r"../tissues/H3K4me3/ENCFF503TXI.bed"
PATH_TO_SAVE_LEARN = r"../gen_samples_data/H3K4me3/train_data/H3K4me3_learn.fasta"
PATH_TO_SAVE_TEST = r"../gen_samples_data/H3K4me3/train_data/H3K4me3_test.fasta"


# ----- Service functions -----


def update_sequence(postfix):
    fin = open(PATH_TO_HG + postfix + ".fasta", 'r')
    seq = "".join(str(x)[:-1] for x in fin.readlines())
    seq = seq[len("assembly") + seq.find("assembly"):]
    fin.close()
    return seq


# ----- Main part -----


if __name__ == '__main__':
    seed(10)

    data_narrowPeaks = pd.read_csv(PATH_TO_TISSUE_FILE, sep='\t', header=None)
    data_narrowPeaks = data_narrowPeaks.sort_values(0, ignore_index=True)

    print("Number of sequences:")
    print(len(data_narrowPeaks))
    print("Number of sequences in chr1:")
    print(len(data_narrowPeaks[data_narrowPeaks[0] == "chr1"]))

    # --- Getting random sequences out of the tissue ---

    indexes_learn = sorted(sample(list(data_narrowPeaks.index), 20000))

    indexes_test = []
    for i in range(len(data_narrowPeaks)):
        if i not in indexes_learn:
            indexes_test.append(i)

    indexes_test = sorted(sample(indexes_test, 3000))

    # Тестируем
    for i in range(len(indexes_learn) - 1):
        if indexes_learn[i] + 1 != indexes_learn[i + 1]:
            print("True")
            break
    for i in indexes_test:
        if i in indexes_learn:
            print("False")
            break
    print(len(indexes_learn) == 20000)
    print(len(indexes_test) == 3000)

    # --- Preparing learn file ---

    with open(PATH_TO_SAVE_LEARN, 'w') as fout:
        postfix = "chr1"
        seq = ""
        seq = update_sequence(postfix)
        for i in indexes_learn:
            if postfix != data_narrowPeaks[0][i]:
                postfix = data_narrowPeaks[0][i]
                seq = update_sequence(postfix)

            middle = (int(data_narrowPeaks[1][i]) + int(data_narrowPeaks[2][i])) // 2
            left = max(middle - 500, 0)
            right = min(middle + 500, len(seq))

            fout.write(">" + postfix + ":" + str(left) + "-" + str(right) + "\n")
            fout.write(seq[left:right] + "\n")

    # --- Preparing test file ---

    with open(PATH_TO_SAVE_TEST, 'w') as fout:
        postfix = "chr1"
        seq = ""
        seq = update_sequence(postfix)
        for i in indexes_test:
            if postfix != data_narrowPeaks[0][i]:
                postfix = data_narrowPeaks[0][i]
                seq = update_sequence(postfix)

            middle = (int(data_narrowPeaks[1][i]) + int(data_narrowPeaks[2][i])) // 2
            left = max(middle - 500, 0)
            right = min(middle + 500, len(seq))

            fout.write(">" + postfix + ":" + str(left) + "-" + str(right) + "\n")
            fout.write(seq[left:right] + "\n")

    # Тестируем
    with open(PATH_TO_SAVE_LEARN, 'r') as test:
        l = test.readlines()
        print(len(l) == 40000)
        b = True
        for i in range(1, 40000, 2):
            if len(l[i]) != 1001:
                b = False
        print(b)

    with open(PATH_TO_SAVE_TEST, 'r') as test:
        l = test.readlines()
        print(len(l) == 6000)
        b = True
        for i in range(1, 6000, 2):
            if len(l[i]) != 1001:
                b = False
        print(b)
