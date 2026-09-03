library(Seurat)
library(SeuratData)  # Only required if you haven't loaded SeuratData previously
library(reticulate)  # Required to work with Python objects in R
library(anndata)

adata <- read_h5ad("/home/rstudio/R/results/Cell/discrete_cancer_stages_annotated.h5ad")
adata

counts <- as.matrix(t(adata$X))
meta.data <- cbind(
  adata$obs["sample_ID"],
  adata$obs["Gender"],
  adata$obs["INSS_stage"],
  adata$obs["discrete_cancer_stage"]
)
# rownames(meta.data) == colnames(as.matrix(t(adata$X)))

robject <- CreateSeuratObject(counts = counts, meta.data = meta.data)

# var.data <- as.data.frame(adata$var["highly_variable"])
# Extract gene names where highly_variable is TRUE
# highly_variable_genes <- rownames(var.data)[var.data$highly_variable]
# Add variable features
# VariableFeatures(robject) <- highly_variable_genes
# Extract the embedding matrix
# embedding_matrix <- as.matrix(adata$obsm[["X_pca_harmony"]])
# rownames(embedding_matrix) <- colnames(robject)
# Create a DimReduc object
# dim_reduc_object <- CreateDimReducObject(
  # embeddings = embedding_matrix, 
  # key = "Harmony_", 
  # assay = DefaultAssay(robject)
# )

# Assign the DimReduc object to the Seurat object with a custom name
# robject@reductions[["harmony"]] <- dim_reduc_object

Idents(robject) <- "discrete_cancer_stage"
levels(robject)
# write to file
saveRDS(robject, file = "/home/rstudio/R/R/data/discrete_cancer_stages_annotated.rds")
