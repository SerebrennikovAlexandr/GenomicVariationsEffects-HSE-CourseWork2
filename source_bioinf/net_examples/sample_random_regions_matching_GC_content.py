import random
import sys
import optparse


#PATH_TO_FASTA_FILE='/home/dsvet/mydata/test_fa.fa'
#PATH_TO_BED_FILE='/home/dsvet/mydata/GENIE3_MSigDB_45TFs/training_data_peaks_extended/ATF2.MsigDB.pos.bed'
#PATH_TO_SAVE_RESULTS='/home/dsvet/mydata/test_fa_sampled.bed'


PATH_TO_FASTA_FILE = '/mnt/data1/lda_model/common_files/hg38.fasta'
# PATH_TO_BED_FILE='/media/seq-srv-05/lcb/dsvet/common_files/bed_files/'
# PATH_TO_SAVE_RESULTS='/media/seq-srv-05/lcb/dsvet/common_files/bed_files_sampled/'


parser = optparse.OptionParser()
parser.add_option("-b", "--bed_file", action="store", type="string",dest="path_to_regions", help='Path to bed file')
parser.add_option("-s", "--save_path", action="store", type="string",dest="path_to_save_sampling", help='Path to save results')
parser.add_option("-f", "--fasta_path", action="store", type="string",dest="path_to_fasta_file", help='path_to_fasta_file')

(options, args) = parser.parse_args()

###----path to bed file with regions the same size you want to sample----###
PATH_TO_BED_FILE = options.path_to_regions
###-----path_to_save_bed file with sampled regions-----###
PATH_TO_SAVE_RESULTS = options.path_to_save_sampling
PATH_TO_FASTA_FILE = options.path_to_fasta_file

tol = 0.01
number_samples_per_region = 50

###---choose genome---###
hg19 = True
dm3_genome = False


def read_bed_file(PATH_TO_BED_FILE):
    bed_file_list = []
    with open(PATH_TO_BED_FILE,'r') as bed_file:
        for line in bed_file:
            line = line.split()
            bed_file_list.append(line)
    return bed_file_list


###----Read fasta file and return dict with keys: fastaID; values:string with fasta sequence----###
def read_fastafile(filename):
    """Read a file in FASTA format
    Arguments:
    filename -- string, the name of the sequence file in FASTA format
    Return:
    list of sequences, list of sequence ids
    """
    id = ''
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
file_to_save = open(PATH_TO_SAVE_RESULTS,'w')


chomosomes_in_fasta = seqID_fasta.keys()

###---Keep only well sequenced chromosomes---###
chomosomes_in_fasta_filterd = [i for i in chomosomes_in_fasta if '_' not in i and i != 'chrM']

###----genarte random region and count G and C content in it-----###
print("Start")
for line in bed_file_list:
    i=0
    while(i<=number_samples_per_region):
        chr = line[0]
        start = int(line[1])
        end = int(line[2])
        dist = end-start

        ###----generate random chr----###
        # if hg19==True:
        #     random_chr=random.randint(1, 22) ###  !!!
        #     chr_rnd='chr' + str(random_chr)
        # if dm3_genome==True:
        #     dm3_chr=['chr2L', 'chr2LHet', 'chr2R', 'chr2RHet',
        #              'chr3L', 'chr3LHet', 'chr3R', 'chr3RHet',
        #              'chr4', 'chrU', 'chrUextra', 'chrX',
        #              'chrXHet', 'chrYHet']
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
