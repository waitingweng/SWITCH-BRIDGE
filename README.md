# SWITCH & BRIDGE: Small-World Informed Topological Clustering Heuristic and Bipartite Reconstruction of the Interstage Dynamics of Gene Expression

## Abstract
Motivation:
Current module detection algorithms struggle to produce reliable results from single-cell data, as their performance is often compromised by data sparsity and high sensitivity to parameter choices. Moreover, current algorithms tend to adopt a growing approach for module detection, which leaves out a considerable number of transiently or weakly correlated genes from modules. These methods are also limited conceptually and typically provide only a static snapshot of gene coexpression. This static view prevents the identification of “transition genes”—the pivotal regulators that mediate the dynamic rewiring of gene networks during processes such as disease progression. Additionally, the current methods for identifying driver genes for disease progression are either DEG based or trajectory inference (TI) based, both of which treat genes independently between stages or over pseudotime and do not consider interactions between genes during stage transition. As a result, a method that uncovers driver genes that rewire modules during stage transition is needed. Pinpointing these key regulators is crucial, as they represent a promising class of therapeutic targets, creating an urgent need for a more robust and dynamic analytical approach. Therefore, new approaches are needed to overcome these limitations and accurately identify crucial molecular players.

Results:
In this study, we developed SWITCH for module detection and BRIDGE, a flexible computational pipeline for identifying transition gene sets and topomodulators with single-cell data. For module detection, we systemically benchmarked SWITCH against three established methods and found that its top-performing ‘flavors’ performed well across key evaluation metrics despite much larger coverage, which unavoidably dilutes module cohesion. Additionally, by constructing bipartite graphs between modules from stages, BRIDGE successfully identified dynamic genes across different disease stages and pinpointed a set of transition genes. Crucially, survival analysis revealed that these transition gene-derived topomodulators were significantly associated with patient outcomes. These findings demonstrate that SWITCH and BRIDGE are powerful tools for detecting not only biologically meaningful gene modules but also clinically relevant therapeutic targets that drive disease progression.

Availability and implementation:
An implementation and the source code of this work are available at https://github.com/waitingweng/SWITCH-BRIDGE.

## Citation

