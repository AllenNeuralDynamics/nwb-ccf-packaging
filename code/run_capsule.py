""" top level run script """

from pathlib import Path
from hdmf_zarr.nwb import NWBZarrIO
import csv
import os
import shutil
from hdmf.common.table import DynamicTable
import numpy as np

data_folder = Path("../data/")
scratch_folder = Path("../scratch/")
results_folder = Path("../results/")

# converts channel name of the form 'LFP1' to idx of the form 0
def channel_name_to_idx(channel_name):
    return int(channel_name[3:])-1


def format_probe_name(probe_name):
    return probe_name[0].upper() + probe_name[1:-1] + probe_name[-1].upper()


def get_channel_idxs(electrodes, probe_names,):
    channel_idxs = {}
    for row in enumerate(electrodes):
        print(row['id'])


def build_ccf_map(probe_csvs):
    ccf_map = {}
    for probe_csv_name in probe_csvs:
        probe_name = format_probe_name(probe_csv_name.stem.split('_')[2])
        probe_info = list(csv.reader(open(probe_csv_name)))
        for channel_id, structure, structure_id, x, y, z, horz, vert, valid, cort_depth in probe_info[1:]:
            ccf_map[probe_name, int(channel_id)] = [structure, x,y,z]

    return ccf_map

def get_new_electrode_colums(nwb, ccf_map):
    locs, xs, ys, zs = [], [], [], []
    for row in nwb.electrodes:
        probe_name = row['group_name'].item()
        channel_id = channel_name_to_idx(row['channel_name'].item())
        try:
            structure, x, y, z = ccf_map[probe_name, channel_id]
        except KeyError:
            raise Exception(f"CCF information for an electrode ({probe_name}, channel {channel_id}) not found. Perhaps not enough CSVs were provided or the given CSVs don't match this session")
        locs.append(structure)
        xs.append(x)
        ys.append(y)
        zs.append(z)
    return locs, xs, ys, zs

def run():
    probe_csvs = [p for p in (data_folder/'ccf').iterdir() if p.name.endswith('.csv')]

    # find base NWB file
    nwb_files = [p for p in (data_folder/'nwb').iterdir() if p.name.endswith(".nwb") or p.name.endswith(".nwb.zarr")]
    assert len(nwb_files) == 1, "Attach one base NWB file data at a time"

    source_path = nwb_files[0]
    destination_dir = '/results/nwb'
    destination_path = os.path.join(destination_dir, os.path.basename(source_path))
    print(source_path, destination_path)
    shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
    nwbfile_input_path = Path(destination_path)

    if nwbfile_input_path.is_dir():
        assert (nwbfile_input_path / ".zattrs").is_file(), f"{nwbfile_input_path.name} is not a valid Zarr folder"
        NWB_BACKEND = "zarr"
        NWB_SUFFIX = ".nwb.zarr"
        io_class = NWBZarrIO
    else:        
        NWB_BACKEND = "hdf5"
        NWB_SUFFIX = ".nwb"
        io_class = NWBHDF5IO
    print(f"NWB backend: {NWB_BACKEND}")

    read_io = io_class(str(nwbfile_input_path), mode='a')
    nwb = read_io.read()


    print(len(nwb.electrodes))
    print(nwb.electrodes[0])
    print(nwb.electrodes[1])
    print(nwb.electrodes[3])

    ccf_map = build_ccf_map(probe_csvs)
    print(ccf_map.keys())
    locs, xs, ys, zs = get_new_electrode_colums(nwb, ccf_map)

    nwb.electrodes.location.data[:] = np.array(locs)
    nwb.electrodes.add_column('x', 'ccf x coordinate', data=xs)
    nwb.electrodes.add_column('y', 'ccf y coordinate', data=ys)
    nwb.electrodes.add_column('z', 'ccf z coordinate', data=zs)

    nwbfile_output_path = results_folder / f"{nwbfile_input_path.stem}.nwb"
    with io_class(str(nwbfile_output_path), "w") as export_io:
        export_io.export(src_io=read_io, nwbfile=nwb, write_args={'link_data': False})
    print(f"Done writing {nwbfile_output_path}")


if __name__ == "__main__": run()