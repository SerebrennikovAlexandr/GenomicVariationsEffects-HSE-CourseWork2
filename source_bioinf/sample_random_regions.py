import random
import sys
import optparse


PATH_TO_FASTA_FILE = '../hg19_human/hg19.fasta'
PATH_TO_BED_FILE = '../tissue/ENCFF503TXI.bed'
PATH_TO_SAVE_RESULTS = '../random_sample.bed'
tol = 0.01
number_samples_per_region = 50


def read_bed_file(PATH_TO_BED_FILE):
    bed_file_list = []
    with open(PATH_TO_BED_FILE,'r') as bed_file:
        for line in bed_file:
            line = line.split()
            bed_file_list.append(line)
    return bed_file_list


# ----Read fasta file and return dict with keys: fastaID; values: string with fasta sequence---- #
def read_fastafile(filename):
    """
    Read a file in FASTA format
    Arguments:
    filename -- string, the name of the sequence file in FASTA format
    Return:
    list of sequences, list of sequence ids
    """
    ids = []
    seqs = []
    seq = []
    with open(filename, 'r') as fh:
        for line in fh:
            if line[0] == '>':
                ids.append(line[1:].rstrip('\n'))
                if seq != []:
                    seqs.append("".join(seq))
                seq = []
            else:
                seq.append(line.rstrip('\n').upper())
        if seq != []:
            seqs.append("".join(seq))
        seqID_fasta = dict(zip(ids,seqs))
    return seqID_fasta


"""
    Main block
"""
bed_file_list = read_bed_file(PATH_TO_BED_FILE)
seqID_fasta = read_fastafile(PATH_TO_FASTA_FILE)

sampled_sequences = []

file_to_save = open(PATH_TO_SAVE_RESULTS, 'w')

chomosomes_in_fasta = seqID_fasta.keys()

# Keep only well sequenced chromosomes
chomosomes_in_fasta_filterd = [i for i in chomosomes_in_fasta if '_' not in i and i != 'chrM']

# Generate random region and count G and C content in it
print("Start")
for line in bed_file_list:
    i = 0
    while i <= number_samples_per_region:
        chr = line[0]
        start = int(line[1])
        end = int(line[2])
        dist = end - start

        chr_rnd = random.choice(chomosomes_in_fasta_filterd)
        chr_length = int(len(seqID_fasta[chr_rnd]))
        rnd_pos = random.randint(0, chr_length-dist+1)
        rnd_start = rnd_pos
        rnd_end = rnd_pos + dist

        ###-----calculate G and C content for initial sequence
        sequenceINIT = seqID_fasta[chr][start:end]
        G_init = sequenceINIT.count("G")
        C_init = sequenceINIT.count("C")
        GC_perc_init = (G_init + C_init)/float(end-start)

        ###----Calculate G an C content for random sequence----###
        sequenceRND = seqID_fasta[chr_rnd][rnd_start:rnd_end]
        if "N" not in sequenceRND:
            G_rnd = sequenceRND.count("G")
            C_rnd = sequenceRND.count("C")
            GC_perc_rnd = (G_rnd + C_rnd)/float(rnd_end-rnd_start)
            if abs(GC_perc_init-GC_perc_rnd) < tol:
                i = i + 1
                file_to_save.write(chr_rnd + "\t" + str(rnd_start) + "\t" + str(rnd_end) + "\n")
        else:
            continue

file_to_save.close()
print("Finished")
