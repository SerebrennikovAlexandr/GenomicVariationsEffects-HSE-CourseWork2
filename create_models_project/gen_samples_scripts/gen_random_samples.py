import pandas as pd
from random import sample, randint, seed


# ----- Settings part -----


PATH_TO_HG = r"../../hg_assemblies/hg19/hg19_"
PATH_TO_RANDOM_SAMPLE_FILE = r"../gen_samples_data/H3K4me3/random_sample_intersected.bed"
PATH_TO_SAVE_RANDOM_LEARN = r"../gen_samples_data/H3K4me3/train_data/H3K4me3_random_learn.fasta"
PATH_TO_SAVE_RANDOM_TEST = r"../gen_samples_data/H3K4me3/train_data/H3K4me3_random_test.fasta"


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

    random_sample = pd.read_csv(PATH_TO_RANDOM_SAMPLE_FILE, sep='\t', header=None)
    random_sample = random_sample.sort_values(0, ignore_index=True)

    # --- Getting random sequences ---

    indexes_random_learn = sorted(sample(list(random_sample.index), 60000))

    indexes_random_test = []
    while len(indexes_random_test) < 9000:
        ind = randint(0, len(random_sample) - 1)
        if ind not in indexes_random_test and ind not in indexes_random_learn:
            indexes_random_test.append(ind)

    indexes_random_learn = sorted(indexes_random_learn)
    indexes_random_test = sorted(indexes_random_test)

    # --- Testing ---

    for i in range(len(indexes_random_learn) - 1):
        if indexes_random_learn[i] + 1 != indexes_random_learn[i + 1]:
            print("True")
            break
    for i in range(len(indexes_random_test) - 1):
        if indexes_random_test[i] + 1 != indexes_random_test[i + 1]:
            print("True")
            break
    for i in indexes_random_learn:
        if i in indexes_random_test:
            print("False")
            break
    print(len(indexes_random_learn) == 60000)
    print(len(indexes_random_test) == 9000)

    # --- Preparing random learn file ---

    with open(PATH_TO_SAVE_RANDOM_LEARN, 'w') as fout:
        postfix = "chr1"
        seq = ""
        seq = update_sequence(postfix)
        for i in indexes_random_learn:
            if postfix != random_sample[0][i]:
                postfix = random_sample[0][i]
                seq = update_sequence(postfix)

            middle = (int(random_sample[1][i]) + int(random_sample[2][i])) // 2
            left = max(middle - 500, 0)
            right = min(middle + 500, len(seq))

            fout.write(">" + postfix + ":" + str(left) + "-" + str(right) + "\n")
            fout.write(seq[left:right] + "\n")

    # --- Preparing random test file ---

    with open(PATH_TO_SAVE_RANDOM_TEST, 'w') as fout:
        postfix = "chr1"
        seq = ""
        seq = update_sequence(postfix)
        for i in indexes_random_test:
            if postfix != random_sample[0][i]:
                postfix = random_sample[0][i]
                seq = update_sequence(postfix)

            middle = (int(random_sample[1][i]) + int(random_sample[2][i])) // 2
            left = max(middle - 500, 0)
            right = min(middle + 500, len(seq))

            fout.write(">" + postfix + ":" + str(left) + "-" + str(right) + "\n")
            fout.write(seq[left:right] + "\n")

    # --- Testing ---

    with open(PATH_TO_SAVE_RANDOM_LEARN, 'r') as test:
        l = test.readlines()
        print(len(l) == 120000)
        b = True
        for i in range(1, 120000, 2):
            if len(l[i]) != 1001:
                b = False
        print(b)

    with open(PATH_TO_SAVE_RANDOM_TEST, 'r') as test:
        l = test.readlines()
        print(len(l) == 18000)
        b = True
        for i in range(1, 18000, 2):
            if len(l[i]) != 1001:
                b = False
        print(b)
