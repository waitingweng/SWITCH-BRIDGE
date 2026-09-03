library(dplyr)
library(Seurat)
library(patchwork)
library(harmony)

setwd("/home/rstudio/R/hdWGCNA")
robject <- readRDS('./data/discrete_cancer_stages_annotated.rds')
robject

# The [[ operator can add columns to object metadata. This is a great place to stash QC stats
robject[["percent.mt"]] <- PercentageFeatureSet(robject, pattern = "^MT-")

robject <- NormalizeData(robject, normalization.method = "LogNormalize", scale.factor = 10000)

robject <- FindVariableFeatures(robject, selection.method = "vst", nfeatures = 2500)

# Identify the 10 most highly variable genes
top10 <- head(VariableFeatures(robject), 10)

# plot variable features with and without labels
plot1 <- VariableFeaturePlot(robject)
plot2 <- LabelPoints(plot = plot1, points = top10, repel = TRUE)
plot1 + plot2

all.genes <- rownames(robject)
robject <- ScaleData(robject, features = all.genes, vars.to.regress = "percent.mt")

robject <- RunPCA(robject, features = VariableFeatures(object = robject))

# Examine and visualize PCA results a few different ways
print(robject[["pca"]], dims = 1:5, nfeatures = 5)
VizDimLoadings(robject, dims = 1:2, reduction = "pca")
DimPlot(robject, reduction = "pca") + NoLegend()
DimHeatmap(robject, dims = 1, cells = 500, balanced = TRUE)
ElbowPlot(robject, ndims = 50)

robject <- RunHarmony(robject, "PatientID", plot_convergence = TRUE)

robject <- RunUMAP(robject, dims = 1:20, reduction = 'harmony', n.neighbors=15)
# note that you can set `label = TRUE` or use the LabelClusters function to help label
# individual clusters
DimPlot(robject, reduction = "umap")


saveRDS(robject, file = "hdWGCNA.rds")
