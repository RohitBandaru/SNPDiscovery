import random
import numpy as np
import pysam
import sys
#samtools mpileup ../data/HG00096.chrom20.ILLUMINA.bwa.GBR.low_coverage.20120522.sam 20:1000000-1100000

txt_file = open('../data/out.sam')
lines = txt_file.readlines()

n_read = 0
start, end = 1000000, 1100000

n_pts = end - start + 1

data = np.zeros((n_pts, 4))
'''
features
1. number of reads aligned
2. proportion match
3. proportion mismatch
4. average quality
'''

print("number of lines: " + str(len(lines)))
for i, row in enumerate(lines):
    if(row[0]!='@'):
        read = row.split()

        qname = read[0]
        flag = read[1]
        rname = read[2]
        pos = int(read[3])
        mapq = float(read[4])
        cigar = read[5]
        rnext = read[6]
        pnext = read[7]
        tlen = read[8]
        seq = read[9]
        qual = read[10]

        # unroll cigar string
        num = ""
        new_cigar = ""
        for c in cigar:
            if(c.isdigit()):
                num += c
            else:
                if(num==""):
                    new_cigar += c
                else:
                    for i in range(int(num)):
                        new_cigar += c
                    num = ""
        start_read = pos - start
        for c in new_cigar:
            if(start_read >= 0 and start_read < n_pts):
                data[start_read,0] += 1

                if(c=="M"):
                    data[start_read,1] += 1
                elif(c=="X"):
                    data[start_read,2] += 1
                data[start_read,3] += mapq
            start_read += 1

        n_read += 1

print("postprocess")
post = data[:,1:]/data[:,0].reshape(n_pts,1)
post[np.isnan(post)]= 0
data[:,1:] = post

# find snps
labels = np.zeros(n_pts)
txt_file = open('../data/my-var-final.vcf')
lines = txt_file.readlines()
for i, row in enumerate(lines):
    if(row[0]!='#'):
        row = row.split()
        pos = int(row[1])
        labels[pos-start] = 1

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

y_pred = clf.predict(X_train)
print(classification_report(y_train, y_pred))

