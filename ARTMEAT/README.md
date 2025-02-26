# BiaxialDeliMeat
Citation: "Biaxial testing and sensory texture perception of plant-based and animal deli meat" https://www.biorxiv.org/content/10.1101/2025.02.19.639170v1

### CANN Input & Code
Input: Biaxial extension with five loading modes for 4 plant-based deli meats and 4 animal deli meats. Loading modes are strip-y, off-y, equibiaxial, off-x, and strip-x.

BiaxCANN_Isotropic.ipynb: Jupyter notebook to run invariant-based constitutive artifical neural network with 8 terms

### Least Squares Regression
1. LSregression_LoadingModeVariation.py: uses data from MeanData folder, fits neo Hooke and Mooney Rivlin models to the mean stress-stretch curves for each meat and calculates R2 for the 5 loading modes
2. LSregression_SampleVariation.py: uses data from SampleData folder, fits neo Hooke and Mooney Rivlin models to each sample individually and reports individual parameters

### Table 2: Full Data
DeliMeatBiaxData.xlsx contains 21 datapoints and 3 significant digits after the decimal

# MechSignature
Citation: "The mechanical and sensory signature of plant-based and animal meat" [https://doi.org/10.1101/2024.04.25.591207](https://doi.org/10.1038/s41538-024-00330-6)

### Input: animal and plant-based meat data
Uniaxial tension/compression and simple shear data for the 5 plant-based and 3 animal meat products analyzed in this paper

### Code
1. CANN4ArtMeat.py: main code to run the eight term invariant-based model
2. models_artmeat_inv.py: build the invariant-based network
3. plottingArtMeat.py: plot the results

# DiscoveringMechanics
Citation: "Discovering the mechanics of artificial and real meat" https://doi.org/10.1016/j.cma.2023.116236

### Input: artifical and real meat data
Uniaxial tension/compression and simple shear data for artifical chick'n, real chicken, and tofurky deli slices

### Code
1. CANN4ArtMeat.py: main code, designed for users to easily switch model type and regularization penalty and type (L1 vs. L2)
2. models_artmeat_stretch_inv.py: build the CANN principal-stretch-based, Valanis-Landel type, and invariant-based networks
3. plottingArtMeat.py: plot the results

### Revision Summary
We have corrected two coding mistakes that had small effects on Table 4, Figures 8, 9, 10, 11, and the text starting on page 18 bottom; the updated table, figures, and text are in RevisedArtMeatPaper.pdf and the corrected code is in this GitHub folder; the main findings of this study are not affected by these updates.
