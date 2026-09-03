library(Seurat)
library(SeuratData)  # Only required if you haven't loaded SeuratData previously
library(reticulate)  # Required to work with Python objects in R
library(anndata)


adata <- read_h5ad("/home/rstudio/R/results/discrete_cancer_stages_annotated.h5ad")
robject <- CreateSeuratObject(counts = as.matrix(t(adata$X)), meta.data = adata$obs, features = adata$var)
Idents(robject) <- "discrete_cancer_stage"
levels(robject)
# write to file
saveRDS(robject, file = "/home/rstudio/R/R_results/discrete_cancer_stages_annotated.rds")



# # Add additional information to the Seurat object
# robject$obsm$X_pca <- adata$obsm$X_pca
# robject$obsm$X_pca_harmony <- adata$obsm$X_pca_harmony
# pbmc$obsm$X_umap <- adata$obsm$X_umap
# pbmc$unstacked$INSS_stage_colors <- adata$uns$INSS_stage_colors
# pbmc$unstacked$hvg <- adata$uns$hvg
# pbmc$unstacked$leiden <- adata$uns$leiden
# pbmc$unstacked$leiden_colors <- adata$uns$leiden_colors
# pbmc$unstacked$leiden_sizes <- adata$uns$leiden_sizes
# pbmc$unstacked$log1p <- adata$uns$log1p
# pbmc$unstacked$neighbors <- adata$uns$neighbors
# pbmc$unstacked$paga <- adata$uns$paga
# pbmc$unstacked$pca <- adata$uns$pca
# pbmc$unstacked$sample_ID_colors <- adata$uns$sample_ID_colors
# pbmc$unstacked$umap <- adata$uns$umap