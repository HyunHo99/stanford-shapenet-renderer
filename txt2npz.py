import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--txtfile', type=str, required=True)
parser.add_argument('--outdir', type=str, default="./")

args = parser.parse_args()

output_file = args.outdir + "cameras.npz"

file = open(args.txtfile, 'r')
cameras={}
lines = file.readlines()
for i in range(0, len(lines), 2):
    key = lines[i].strip()
    val = lines[i+1].strip()
    val = val.replace("[","").replace("]","").split(",")
    mat = [float(i) for i in val]
    mat = np.array(mat).reshape(3,4)
    mat = np.concatenate((mat,[[0,0,0,1]]),axis=0)
    cameras[key] = mat

np.savez(output_file, **cameras)
file.close()