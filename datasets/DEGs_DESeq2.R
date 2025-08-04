# Install DESeq2 package
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("DESeq2")

# Load DESeq2 package
library(DESeq2)

# Read RNA-Seq expression data (original count matrix)
rna_data <- read.csv("../datasets_csv/lusc/Gene_Total/0_train.csv", row.names = 1)
head(rna_data)

# Read sample information
sample_info <- read.csv("../datasets_csv/lusc/Info_Sample/0_info.csv")

# Create DESeq2 dataset
dds <- DESeqDataSetFromMatrix(countData = rna_data,
                              colData = sample_info,
                              design = ~ condition)

# Differential expression analysis
dds <- DESeq(dds)
res <- results(dds)
head(res)

# Filter significantly differential expressed genes
degs <- res[which(res$padj < 0.05 & abs(res$log2FoldChange) > 0.5), ]

# Calculate composite score using Z-score normalization
degs$z_fdr <- scale(-log10(degs$padj))
degs$z_logfc <- scale(abs(degs$log2FoldChange))
degs$combined_score <- degs$z_fdr + degs$z_logfc  # Equal-weight integration

# Select Top-N genes based on composite score
N <- 200  # Number of genes to select
top_degs <- degs[order(degs$combined_score, decreasing = TRUE), ]
top_degs <- top_degs[1:N, ]

# Create result dataframe (including composite scores)
top_degs_df <- data.frame(
  Hugo_Symbol = rownames(top_degs),
  log2FC = top_degs$log2FoldChange,
  baseMean = top_degs$baseMean,
  pvalue = top_degs$pvalue,
  FDR = top_degs$padj,
  z_FDR = top_degs$z_fdr,
  z_log2FC = top_degs$z_logfc,
  combined_score = top_degs$combined_score
)

# Save results to CSV file
write.csv(top_degs_df, "../datasets_csv/lusc/Gene_DEGs/0_train.csv", row.names = FALSE)

# Visualize composite score distribution (optional)
png("../datasets_csv/lusc/Gene_DEGs/0_train.png", width=800, height=600)
hist(degs$combined_score, breaks=50, main="Distribution of Combined Significance Scores",
     xlab="Combined Score (Z-FDR + Z-log2FC)", col="skyblue", border="white")
abline(v=top_degs$combined_score[N], col="red", lty=2, lwd=2)
dev.off()

# Volcano plot visualization (highlight top N genes)
png("../datasets_csv/lusc/Gene_DEGs/0_Volcano.png", width=1000, height=800)
with(res, plot(log2FoldChange, -log10(padj), pch=20, 
               main="Volcano Plot with Top-K Genes",
               xlab="log2 Fold Change", ylab="-log10 FDR"))
with(subset(res, padj < 0.05 & abs(log2FoldChange) > 1), 
     points(log2FoldChange, -log10(padj), pch=20, col="blue"))
with(top_degs, points(log2FoldChange, -log10(padj), pch=20, col="red"))
dev.off()