# Input subfolder: raw_data
Contains all raw experimental data and sample dimension measurements for all experiments conducted in tension, shear, confined compression, and unconfined compression. 

# Input file: \*_raw_data.xlsx
Contains the processed data for tension, unconfined compression, and shear with data columns for time (seconds), deformation (stretch or shear strain), and stress (axial piola stress P11 or shear piola stress P12). 

# Input file: FoamData.xlsx
Contains all the averaged elastic experimental data for tension, unconfined compression, and shear. The data columns are deformation (stretch or shear strain) and stress (axial piola stress P11 or shear piola stresss P12, average of loading and unloading)

# elastic-cann
Citation: "Discovering the mechanics of ultra-low density elastomeric foams in elite-level racing shoes." https://arxiv.org/abs/2602.12694
1. main.py: main code
2. models.py: build the CANN model with options to enable or disable single invariant terms, mixed invariant terms, and principal stretch terms. 
3. plotting.py: Create plots of discovered models predictions and contributions of different terms
4. util.py: Utility functions to parse data, setup training, and build a complete constitutive neural network based on the Psi model (model which maps invariants to strain energy).
5. preprocessing.py: Standalone file which takes raw data from experiments and outputs processed \*_raw_data.xlsx and FoamData.xlsx, as well as plots and tables based on the raw data. 

