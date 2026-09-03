library(dplyr)
library(Seurat)
library(patchwork)
library(harmony)

setwd("/home/rstudio/R/hdWGCNA")
seurat_obj <- readRDS('./data/discrete_cancer_stages_annotated.rds')
seurat_obj

# The [[ operator can add columns to object metadata. This is a great place to stash QC stats
seurat_obj[["percent.mt"]] <- PercentageFeatureSet(seurat_obj, pattern = "^MT-")

seurat_obj <- NormalizeData(seurat_obj, normalization.method = "LogNormalize", scale.factor = 10000)

seurat_obj <- FindVariableFeatures(seurat_obj, selection.method = "vst", nfeatures = 2500)

# Identify the 10 most highly variable genes
top10 <- head(VariableFeatures(seurat_obj), 10)

# plot variable features with and without labels
plot1 <- VariableFeaturePlot(seurat_obj)
plot2 <- LabelPoints(plot = plot1, points = top10, repel = TRUE)
plot1 + plot2

all.genes <- rownames(seurat_obj)
seurat_obj <- ScaleData(seurat_obj, features = all.genes, vars.to.regress = "percent.mt")

seurat_obj <- RunPCA(seurat_obj, features = VariableFeatures(object = seurat_obj))

# Examine and visualize PCA results a few different ways
print(seurat_obj[["pca"]], dims = 1:5, nfeatures = 5)
VizDimLoadings(seurat_obj, dims = 1:2, reduction = "pca")
DimPlot(seurat_obj, reduction = "pca") + NoLegend()
DimHeatmap(seurat_obj, dims = 1, cells = 500, balanced = TRUE)
ElbowPlot(seurat_obj, ndims = 50)

seurat_obj <- RunHarmony(seurat_obj, "sample_ID", plot_convergence = TRUE)

seurat_obj <- RunUMAP(seurat_obj, dims = 1:50, reduction = 'harmony', n.neighbors=15)
# note that you can set `label = TRUE` or use the LabelClusters function to help label
# individual clusters
DimPlot(seurat_obj, reduction = "umap")


saveRDS(seurat_obj, file = "./data/discrete_cancer_stages_annotated_hdWGCNA.rds")
