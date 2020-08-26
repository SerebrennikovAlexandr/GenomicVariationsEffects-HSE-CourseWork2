# ----- Settings part -----


PATH_TO_ASSEMBLY = '../hg19_human/hg19.fa'
PATH_TO_FASTA_FILE = '../hg19_human/hg19.fasta'


# ----- Service functions -----


def format_assembly():
    with open(PATH_TO_ASSEMBLY, 'r') as file:
        fin = open(PATH_TO_FASTA_FILE, 'w')
        for line in file:
            if line[0] == '>':
                fin.write(line)
            else:
                fin.write(line.upper())
        fin.close()


# ----- Main part -----


if __name__ == '__main__':
    # Форматирование файла сборки генома hg19
    format_assembly()
