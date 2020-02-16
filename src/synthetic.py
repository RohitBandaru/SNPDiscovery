import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def synthDNA(len, het_snp_freq, hom_snp_freq):
    ref_dna = ""
    het_snps = {}
    hom_snps = {}
    for i in range(len):
        ref_base = random.choice("cgta")
        ref_dna += ref_base

        if(random.random() < het_snp_freq):
            # simulate heterozygous SNP
            minor_base = random.choice("cgta".replace(ref_base, ""))
            minor_allele_freq = np.random.uniform(0.01,0.5)
            het_snps[i] = {"base": minor_base, "minor_freq": minor_allele_freq}
        elif(random.random() < hom_snp_freq):
            # simulate homozygous SNP
            minor_base = random.choice("cgta".replace(ref_base, ""))
            hom_snps[i] = minor_base

    return ref_dna, het_snps, hom_snps

def gen_reads(ref_dna, het_snps, hom_snps, n_reads, read_length_range):
    min_len, max_len = read_length_range
    reads = []

    for i in range(n_reads):
        start = random.randint(0, len(ref_dna) - max_len)
        read_length = random.randint(min_len, max_len)
        ref_read = ref_dna[start:start+read_length]

        q_r = np.random.choice([.5, .1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
            p=[.5,.25,.125,.0625,.03125,0.015625,0.0078125,0.0078125])

        read = ""
        for j, b in enumerate(ref_read):
            base_index = j + start

            # heterozygous SNP
            if base_index in het_snps:
                if(random.random() < het_snps[base_index]["minor_freq"]):
                    b = het_snps[base_index]["base"]

            # homozygous SNP
            if base_index in hom_snps:
                b = hom_snps[base_index]

            # sequencing error
            if(random.random() < q_r):
                read += random.choice("cgta".replace(b, ""))
            else:
                read += b

        reads.append((start, read))

    return reads


def gen_data(ref_dna, reads, het_snps, hom_snps):
    n = len(ref_dna)
    base_to_int = {b:i for i, b in enumerate("cgta")}
    labels = np.zeros(len(ref_dna))
    for i in range(len(ref_dna)):
        if(i in het_snps):
            labels[i] = 1
        elif(i in hom_snps):
            labels[i] = 2

    counts = np.zeros((n,4))
    for start, read in reads:
        for i, base in enumerate(read):
            counts[start+i, base_to_int[base]] += 1

    '''
    features
    1. proportion of ref
    2. highest proportion
    3. 2nd high proportion
    4. number of aligned reads
    '''
    data = np.zeros((n,4))
    for i in range(n):
        data[i,0] = counts[i,base_to_int[ref_dna[i]]]
        data[i,1:3] = np.flipud(np.sort(counts[i]))[:2]
        data[i, 3] = np.sum(counts[i])

    return data, labels

def experiment1():
    # copied from jupyter notebook
    ref_dna, het_snps, hom_snps = synthDNA(100000, 1e-3, 5e-4)
    n = len(ref_dna)
    reads = gen_reads(ref_dna, het_snps, hom_snps, 500000, (10,100))
    data, labels = gen_data(ref_dna, reads, het_snps, hom_snps)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    y_pred = clf.predict(X_train)
    print(classification_report(y_train, y_pred))

def experiment2():
    ref_dna, het_snps, hom_snps = synthDNA(100000, 1e-3, 5e-4)
    reads = gen_reads(ref_dna, het_snps, hom_snps, 100000, (10,100))
    data, labels = gen_data(ref_dna, reads, het_snps, hom_snps)

    for i in range(4):
        ref_dna, het_snps, hom_snps = synthDNA(100000, 1e-3, 5e-4)
        n = len(ref_dna)
        reads = gen_reads(ref_dna, het_snps, hom_snps, 100000, (10,100))
        data_ind, labels_ind = gen_data(ref_dna, reads, het_snps, hom_snps)
        data = np.concatenate([data,data_ind],axis=0)
        labels = np.concatenate([labels, labels_ind])
    X_train, y_train = data, labels

    ref_dna, het_snps, hom_snps = synthDNA(100000, 1e-3, 5e-4)
    reads = gen_reads(ref_dna, het_snps, hom_snps, 100000, (10,100))
    X_test, y_test = gen_data(ref_dna, reads, het_snps, hom_snps)

    clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    pass

