# NWB-Packaging-CCF

This Capsule is intended to take an existing NWB file and CSVs containing probe CCF information, and modify the NWB *electrodes* table to include this CCF information. In addition to requiring the NWB to exist in the data directory, it requires a **CCF* asset containing a .CSV file for each probe (usually 6). 

Each CSV is expected to contain the probe name within its filename, of the form `probeX` and to have the following columns in this order:
**channel_id, structure_acronym, structure_id, x_coord, y_coord, z_coord, horizontal_position, vertical_position, is_valid, cortical_depth**.

Importantly, all probes whose that were used to package the inputted NWB file and produce the electrodes table, must have .CSVs present in the input! Otherwise, updating the electrodes table will fail because some electrodes rows will not have coordinate information to add.

The NWB File may be in Zarr or HDMF format.