# Install limma package
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("limma")

# Load limma package
library(limma)

# Read RNA-Seq expression data file (Z-score data)
rna_data <- read.csv("../datasets_csv/lusc/Gene_Total/0_train.csv", row.names = 1)

# Read sample_info.csv containing sample grouping information (condition)
sample_info <- read.csv("../datasets_csv/lusc/Info_Sample/0_info.csv")

# Display first few rows of data
head(rna_data)
head(sample_info)

# Create design matrix based on 'condition' column for grouping
design <- model.matrix(~ condition, data = sample_info)

# Fit linear model:Perform differential expression analysis using limma
fit <- lmFit(rna_data, design)      
# Apply Bayesian statistics for hypothesis testing
fit <- eBayes(fit)                   

# Retrieve differential expression results
# 'condition' column has two categories: long_survival, short_survival
results <- topTable(fit, adjust="fdr", coef=2, number=Inf)

# Display top differential expression results
head(results)

# Filter significantly differential expressed genes
degs <- results[which(results$adj.P.Val < 0.05 & abs(results$logFC) > 0.5), ]

# Calculate composite score using Z-score normalization
degs$z_fdr <- scale(-log10(degs$adj.P.Val))
degs$z_logfc <- scale(abs(degs$logFC))
degs$combined_score <- degs$z_fdr + degs$z_logfc  # Equal-weight integration

# Select Top-N genes based on composite score
N <- 200  # Number of genes to select
top_degs <- head(degs[order(degs$combined_score, decreasing = TRUE), ], N)

# Display filtered significant genes
head(top_degs)

# Save results to CSV file
write.csv(top_degs, "../datasets_csv/lusc/Gene_DEGs/0_train.csv", row.names = TRUE)

# Visualize composite score distribution (optional)
png("../datasets_csv/lusc/Gene_DEGs/0_train.png", width=800, height=600)
hist(degs$combined_score, breaks=50, main="Distribution of Combined Significance Scores", 
     xlab="Z-score Combined Score", col="skyblue", border="white")
abline(v=degs$combined_score[rownames(top_degs)], col="red", lty=2)
dev.off()