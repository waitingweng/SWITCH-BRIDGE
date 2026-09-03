library(Seurat)
library(SeuratData)  # Only required if you haven't loaded SeuratData previously
library(reticulate)  # Required to work with Python objects in R
library(anndata)

adata <- read_h5ad("/home/rstudio/R/COVID/COVID_macrophage_scVI.h5ad")
adata

counts <- as.matrix(t(adata$X))
meta.data <- cbind(
  adata$obs["PatientID"],
  adata$obs["CoVID-19 severity"],
  adata$obs["Age"],
  adata$obs["Sex"],
  adata$obs['batch']
)
# rownames(meta.data) == colnames(as.matrix(t(adata$X)))

robject <- CreateSeuratObject(counts = counts, meta.data = meta.data)
# harmony <- as.matrix(adata$obsm['X_pca_harmony']$X_pca_harmony)
# rownames(harmony) <- paste0("Cell_", 1:56861)
# robject@reductions[['harmony']] <- CreateDimReducObject(embeddings = harmony, key = "Harmony_", assay = DefaultAssay(robject))
# 
# scvi <- as.matrix(adata$obsm['X_scVI']$X_scVI)
# rownames(scvi) <- paste0("Cell_", 1:56861)
# robject@reductions[['scVI']] <- CreateDimReducObject(embeddings = scvi, key = "scvi_", assay = DefaultAssay(robject))
# 
# pca <- as.matrix(adata$obsm['X_pca']$X_pca)
# rownames(pca) <- paste0("Cell_", 1:56861)
# robject@reductions[['pca']] <- CreateDimReducObject(embeddings = pca, key = "pca_", assay = DefaultAssay(robject))
# 
# umap <- as.matrix(adata$obsm['X_umap']$X_umap)
# rownames(umap) <- paste0("Cell_", 1:56861)
# robject@reductions[['umap']] <- CreateDimReducObject(embeddings = umap, key = 'umap_', assay = DefaultAssay(robject))


Idents(robject) <- "CoVID-19 severity"
levels(robject)
# write to file
saveRDS(robject, file = "/home/rstudio/R/COVID/parameter_tuning/hdWGCNA/Cell/Macrophage_so.rds")
