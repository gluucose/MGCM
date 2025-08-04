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

# Extract gene expression matrix and handle missing values
gene_exp <- as.matrix(data[2:nrow(data), 2:201])
gene_exp <- apply(gene_exp, 2, as.numeric)
gene_exp[is.na(gene_exp)] <- 0

# Load WGCNA and configure options
library(WGCNA)
options(stringsAsFactors = FALSE)
allowWGCNAThreads()

# Define candidate minimum module sizes
min_module_sizes <- c(9, 11, 13, 15, 17)

# Initialize results data frame
results <- data.frame(
  ModuleSize = integer(),
  Module = character(),
  GeneCount = integer()
)

# Iterate over different minimum module sizes
for (size in min_module_sizes) {
  # Perform block-wise module detection
  net <- blockwiseModules(
    gene_exp,
    power = 8,              # Fixed soft threshold power
    minModuleSize = size,   # Minimum module size
    mergeCutHeight = 0.25,  # Fixed module merging threshold
    numericLabels = TRUE,   # Use numeric module labels
    pamRespectsDendro = FALSE,
    saveTOMs = TRUE,
    verbose = 3
  )
  
  # Print module distribution for current size
  cat("Module size:", size, "\n")
  print(table(net$colors))
  
  # Store module counts
  module_table <- table(net$colors)
  for (module in names(module_table)) {
    results <- rbind(results, data.frame(
      ModuleSize = size,
      Module = module,
      GeneCount = module_table[module]
    ))
  }
}

# Print summary table
print(results)

# Plot module distribution by minimum module size
library(ggplot2)
ggplot(results, aes(x = factor(ModuleSize), y = GeneCount, fill = Module)) +
  geom_bar(stat = "identity", position = "stack") +
  labs(
    title = "Module Distribution by Minimum Module Size",
    x = "Minimum Module Size",
    y = "Gene Count"
  ) +
  theme_minimal()