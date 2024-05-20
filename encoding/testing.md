testing:
 sequential (concurrent reading then gpu submittance)

time ./host 1 10 696001 10  >/dev/null

real    0m18,410s
user    0m35,451s
sys     0m0,684s

 concurrent (concurrent reading and gpu submittance)
 time ./host 1 10 696001 10  >/dev/null

real    0m17,809s
user    0m34,564s
sys     0m0,679s


concurrent with first kernel execution with 20 batched structs for a single batch (and reading & printing results)
time ./host 1 10 696001 10  > aux

real    0m15,626s
user    0m29,476s
sys     0m0,737s

concurrent with first kernel execution with 20  batched structs (and reading & printing results)
time ./host 1 10 696001 10  > aux

real    2m19,382s
user    2m58,428s
sys     0m12,423s


concurrent with first kernel execution with 20  batched structs (without reading & printing results)


concurrent with first kernel execution with 100 batched structs (and reading &  printing results)
time ./host 1 10 696001 10 1 > aux

real    2m25,748s
user    3m10,429s
sys     0m13,067s


