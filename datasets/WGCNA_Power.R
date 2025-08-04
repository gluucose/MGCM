# Set data and output directories
data_dir <- "../datasets_csv/lusc/Gene_DEGs"
output_dir <- "../datasets_csv/lusc/Info_Params"

# Construct full path to data file
data_file <- file.path(data_dir, "0_train.csv")

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

# Replace missing values with 0
gene_exp[is.na(gene_exp)] <- 0

# Load WGCNA library and configure options
library(WGCNA)
options(stringsAsFactors = FALSE)
allowWGCNAThreads()

# Define candidate powers for soft threshold
powers <- c(1:10, seq(from = 12, to = 20, by = 2))

# Select optimal soft threshold power
sft <- pickSoftThreshold(gene_exp, powerVector = powers, verbose = 5)

# Plot scale independence to determine optimal power
plot(sft$fitIndices$Power, -sign(sft$fitIndices$SFT.R.sq) * sft$fitIndices$SFT.R.sq,
     xlab = "Soft Threshold (power)",
     ylab = "Scale Free Topology Model Fit (signed R^2)",
     type = "n",
     main = "Scale Independence"
)
text(sft$fitIndices$Power, -sign(sft$fitIndices$SFT.R.sq) * sft$fitIndices$SFT.R.sq,
     labels = powers, cex = 0.9, col = "red")
abline(h = 0.9, col = "red")  # Threshold for good fit (R^2 >= 0.9)