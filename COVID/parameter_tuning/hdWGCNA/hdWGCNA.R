# single-cell analysis package
library(Seurat)

# plotting and data science packages
library(tidyverse)
library(cowplot)
library(patchwork)
library(ggplot2)

# co-expression network analysis packages:
library(WGCNA)
library(hdWGCNA)

library(feather)

# using the cowplot theme for ggplot
theme_set(theme_cowplot())

# set random seed for reproducibility
set.seed(12345)

# optionally enable multithreading
enableWGCNAThreads(nThreads = 12)

# load the Zhou et al snRNA-seq dataset
# obj <- readRDS('./R/R/data/Zhou_2020.rds')
setwd("/home/rstudio/R/COVID/parameter_tuning/hdWGCNA/Cell")
seurat_obj <- readRDS('hdWGCNA.rds')

seurat_obj@meta.data$covid <- as.factor(gsub("/.*", "", seurat_obj@meta.data$`CoVID-19 severity`))

seurat_obj <- SetupForWGCNA(
  seurat_obj,
  gene_select = "fraction", # the gene selection approach
  fraction = 0, # fraction of cells that a gene needs to be expressed in order to be included
  wgcna_name = "hdWGCNA" # the name of the hdWGCNA experiment
)

# construct metacells  in each group
seurat_obj <- MetacellsByGroups(
  seurat_obj = seurat_obj,
  group.by = c("PatientID","covid"), # specify the columns in seurat_obj@meta.data to group by
  reduction = 'harmony', # select the dimensionality reduction to perform KNN on
  k = 30, # nearest-neighbors parameter
  min_cells = 50, # maximum number of shared cells between two metacells
  ident.group = "covid" # set the Idents of the metacell seurat object
)

# seurat_obj <- NormalizeMetacells(seurat_obj)
# metacell_obj <- GetMetacellObject(seurat_obj)

seurat_obj <- NormalizeMetacells(seurat_obj)
all.genes <- rownames(seurat_obj)
seurat_obj <- ScaleMetacells(seurat_obj, features=all.genes)
seurat_obj <- RunPCAMetacells(seurat_obj, features=VariableFeatures(seurat_obj))
seurat_obj <- RunHarmonyMetacells(seurat_obj, group.by.vars='PatientID')
seurat_obj <- RunUMAPMetacells(seurat_obj, reduction='harmony', dims=1:50)


saveRDS(seurat_obj, file='hdWGCNA_object.rds')

seurat_obj <- readRDS('hdWGCNA_object.rds')
seurat_obj <- SetDatExpr(
  seurat_obj,
  group_name = "control", # the name of the group of interest in the group.by column
  group.by='covid', # the metadata column containing the cell type info. This same column should have also been used in MetacellsByGroups
  assay = 'RNA', # using RNA assay
  slot = 'data' # using normalized data
)

# Test different soft powers:
seurat_obj <- TestSoftPowers(
  seurat_obj,
  networkType = 'unsigned' # you can also use "signed" or "signed hybrid"
)

# plot the results:
plot_list <- PlotSoftPowers(seurat_obj, point_size = 7, text_size = 5)
# Customize text sizes for each plot in the list
plot_list <- lapply(plot_list, function(p) {
  p + theme(
    axis.title.x = element_text(size = 18),
    axis.title.y = element_text(size = 18),
    axis.text.x = element_text(size = 15),
    axis.text.y = element_text(size = 15),
    # plot.title = element_text(size = 22, hjust = 0.5),  # Center the title
    plot.margin = margin(15, 10, 15, 10)  # Adjust margins (top, right, bottom, left)
  )
})
# assemble with patchwork
combined_plot <- wrap_plots(plot_list, ncol = 2)
# Display the combined plot
print(combined_plot)
# Optionally, save the combined plot
ggsave(filename = "./figures/soft_powers_ctrl.svg", plot = combined_plot, width = 12, height = 10)


# run it just for getting TOM
seurat_obj <- ConstructNetwork(
  seurat_obj,
  tom_name = "control", # Name the TOM file to include deepSplit value
  networkType = "unsigned",
  overwrite_tom = TRUE,
)
# get TOM
TOM <- GetTOM(seurat_obj)
TOM_df <- as.data.frame(TOM)
write_feather(TOM_df, "./TOM/TOM_ctrl.feather")
# write.csv(TOM, file = "./TOM/TOM_4S.csv", row.names = TRUE)

# construct co-expression network:
# Generate deepSplit values from 0~4 with a step of 0.5, the valid values are 0~4
deepSplit_values <- seq(0, 4, by = 0.5)
# Loop over each deepSplit value
for (deepSplit in deepSplit_values) {
  # Construct the network with the current deepSplit value
  seurat_obj <- ConstructNetwork(
    seurat_obj,
    tom_name = paste0('ctrl_deepSplit_', deepSplit), # Name the TOM file to include deepSplit value
    networkType = "unsigned",
    overwrite_tom = TRUE,
    deepSplit = deepSplit,  # Current deepSplit value
    minModuleSize = 4
  )
  
  # Optionally, you can save the Seurat object or results to a file here
  # saveRDS(seurat_obj, file = paste0('seurat_obj_deepSplit_', deepSplit, '.rds'))
  # Open an SVG device
  
  # svg(filename = paste0("./figures/hdWGCNA_severe_dendrogram_", deepSplit, ".svg"), width = 8, height = 6)
  # # Example code if `PlotDendrogram()` returns a ggplot object
  # PlotDendrogram(seurat_obj, main = 'Severe hdWGCNA Dendrogram', 
  #                cex.main = 3, cex.axis = 2, cex.lab = 2)
  # # Close the SVG device
  # dev.off()
  
  # get the module assignment table:
  modules <- GetModules(seurat_obj) %>% subset(module != 'grey')
  # Save the dataframe as a CSV file
  write.csv(modules, file = paste0("./results/module_ctrl_", deepSplit, ".csv"))
}