# Autoencoder
Represent the immune system TCR repertoire and estimate similarity and distance between different repertoires

This analysis was tested using six different datasets:

1. GLIPH (Glanville): TCR sequences were taken from 7 T-cell specificities. Each peptide has corresponding one sample of TCRs with probability of sharing specificity. The TCRs from different individuals were clustered using the GLIPH algorithm.

2. ImmunoMap (Sidhom): CD8 T cells responding to Kb-TRP2, a shared self-peptide tumor antigen, and Kb-SIY, a model foreign-antigen, in 4-5 naïve and tumor-bearing B6 mice.

3. TILs Immunotherapies (Rudqvist): Samples were taken from cohorts of tumor-bearing mice treated with the following immunotherapies: radiotherapy (RT), anti–CTLA-4 (9H10) and a combination of them. 

4. Naïve-memory: Peripheral blood mononuclear cells (PBMC) of three donors were sorted into memory (central memory, CM; effector memory, EM; effector memory RA-expressing revertants, EMRA) and naïve CD4+ and CD8+ populations, alpha and beta chains.

5. Vaccine: Yellow fever vaccine (YFV) responding TCRs in three pairs of identical twin donors (P1, P2, Q1, Q2, S1, S2) at different time-points.

6. Cancer: A collection of TCRs from hosts with any cancer and healthy hosts downloaded from the Immune Epitope Database (IEDB).
