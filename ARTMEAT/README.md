
# DiscoveringMechanics
Citation: "Discovering the mechanics of artificial and real meat" https://doi.org/10.1016/j.cma.2023.116236

## Input: artifical and real meat data
Uniaxial tension/compression and simple shear data for artifical chick'n, real chicken, and tofurky deli slices

## Code
1. CANN4ArtMeat.py: main code, designed for users to easily switch model type and regularization penalty and type (L1 vs. L2)
2. models_artmeat_stretch_inv.py: build the CANN principal-stretch-based, Valanis-Landel type, and invariant-based networks
3. plottingArtMeat.py: plot the results

## Revision Summary
We have corrected two coding mistakes that had small effects on Table 4, Figures 8, 9, 10, 11, and the text starting on page 18 bottom; the updated table, figures, and text are in RevisedArtMeatPaper.pdf and the corrected code is in this GitHub folder; the main findings of this study are not affected by these updates.

# GotMeat?
Citation: "Got meat? The mechanical signature of animal and plant-based meat" https://doi.org/10.1101/2024.04.25.591207

## Input: animal and plant-based meat data
Uniaxial tension/compression and simple shear data for the 5 plant-based and 3 animal meat products analyzed in this paper

## Code
1. CANN4ArtMeat.py: main code to run the eight term invariant-based model
2. models_artmeat_inv.py: build the invariant-based network
3. plottingArtMeat.py: plot the results
