# Gene Co-expression Network Analysis: Calculate Adjacency Matrix with WGCNA

# Set data and output directories
data_dir <- "../datasets_csv/lusc/Gene_DEGs"
output_dir <- "../datasets_csv/lusc/Adj_Matrix"

# Construct full paths
data_file <- file.path(data_dir, "0_train.csv")
output_file <- file.path(output_dir, "0_Adj.csv")

# Parameters for WGCNA
wgcna_power <- 8                  # Soft threshold power
wgcna_minModuleSize <- 9          # Minimum module size
wgcna_mergeCutHeight <- 0.25      # Module merging threshold

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
  cat("Created directory:", output_dir, "\n")
} else {
  cat("Directory exists:", output_dir, "\n")
}

# Check if data file exists and read it
if (file.exists(data_file)) {
  data <- read.csv(data_file, header = FALSE)
  cat("Successfully read data file:", data_file, "\n")
} else {
  stop("Data file not found:", data_file)
}

# Extract gene expression matrix (rows 2-end, columns 2-201)
gene_exp <- as.matrix(data[2:nrow(data), 2:201])

# Ensure genes are rows and samples are columns
# gene_exp <- t(gene_exp)

# Print matrix dimensions
cat("Dimensions of gene expression matrix:", dim(gene_exp), "\n")

# Replace missing values (NA) with 0
gene_exp[is.na(gene_exp)] <- 0

# Load WGCNA library
library(WGCNA)

# Calculate adjacency matrix using the defined power
adjacency <- adjacency(gene_exp, power = wgcna_power)

# Save adjacency matrix to CSV file
write.csv(adjacency, file = output_file, quote = FALSE, row.names = FALSE)
cat("Adjacency matrix saved to:", output_file, "\n")