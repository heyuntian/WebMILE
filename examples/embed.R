# This script uses the library available at https://github.com/YosefLab/Rline.

#loading the project
library(rline)
library("argparse")
parser <- ArgumentParser()
parser$add_argument('--input', default="coarsened_graph.edgelist", help='Path of input graph data.')
parser$add_argument('--output', default="coarsened_embeddings.txt", help='Path of output embeddings.')
parser$add_argument('--embed-dim', type = "integer", default=8, help='Dimensionality of embeddings.')
parser$add_argument('--workers', type = "integer", default=1, help='Number of threads.')
args <- parser$parse_args()

#Initalizing Inputs
raw_data <- read.table(args$input, header=FALSE)
u <- raw_data[,1]
v <- raw_data[,2]
w <- if (ncol(raw_data) == 3) raw_data[,3] else rep(1, nrow(raw_data))
edge_list_df <- data.frame(u, v, w)

#LINE Algorithm
d1 <- trunc(args$embed_dim / 2)
d2 <- args$embed_dim - d1
reconstruct_df <- reconstruct(edge_list_df, max_depth = 2, max_k = 10)
line_one_matrix <- line(reconstruct_df, dim = d1, order = 1, negative = 5, samples = 10, rho = 0.025, threads = args$workers)
line_two_matrix <- line(reconstruct_df, dim = d2, order = 2, negative = 5, samples = 10, rho = 0.025, threads = args$workers)
concatenate_matrix <- concatenate(line_one_matrix, line_two_matrix)
normalize_matrix <- normalize(concatenate_matrix)

#Printing Outputs
normalize_matrix <- normalize_matrix[ order(as.numeric(row.names(normalize_matrix))), ]
write.table(normalize_matrix, file=args$output, row.names=FALSE, col.names=FALSE)