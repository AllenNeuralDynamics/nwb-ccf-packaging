""" top level run script """

import argparse
from pathlib import Path
from hdmf_zarr.nwb import NWBZarrIO
from pynwb import NWBHDF5IO
import csv
import os
import shutil
from hdmf.common.table import DynamicTable
import numpy as np

data_folder = Path("../data/")
scratch_folder = Path("../scratch/")
results_folder = Path("../results/")

# converts channel name of the form 'LFP1' or 'AP1' to idx of the form 0
def channel_name_to_idx(channel_name):
    stripped_name = channel_name.replace('LFP','').replace('AP','')
    try:
        return int(stripped_name)-1
    except:
        raise Exception('Unexpected channel name format')


def probe_name_from_file_name(csv_path):
    probe_name = None
    for string_segment in csv_path.stem.split('_'):
        if string_segment.lower().startswith('probe'):
            probe_name = string_segment
            break
    if not probe_name:
        raise Exception('This file name does not appear to contain a probe name')
    return format_probe_name(probe_name)


def format_probe_name(probe_name):
    return probe_name[0].upper() + probe_name[1:-1] + probe_name[-1].upper()


def build_ccf_map(probe_csvs):
    ccf_map = {}
    for probe_csv_path in probe_csvs:
        print('Reading',probe_csv_path)
        probe_name = probe_name_from_file_name(probe_csv_path)
        probe_info = list(csv.reader(open(probe_csv_path)))
        for channel_id, structure, structure_id, x, y, z, horz, vert, valid, cort_depth in probe_info[1:]:
            print('adding', probe_name, int(channel_id))
            ccf_map[probe_name, int(channel_id)] = [structure, float(x),float(y),float(z)]

    return ccf_map


def get_new_electrode_colums(nwb, ccf_map):
    locs, xs, ys, zs = [], [], [], []
    for row in nwb.electrodes:
        probe_name = format_probe_name(row['group_name'].item())
        channel_id = channel_name_to_idx(row['channel_name'].item())
        try:
            structure, x, y, z = ccf_map[probe_name, channel_id]
        except KeyError:
            raise Exception(f"CCF information for an electrode ({probe_name}, channel {channel_id}) not found. Perhaps not enough CSVs were provided or the given CSVs don't match this session")
        locs.append(structure)
        xs.append(x)
        ys.append(y)
        zs.append(z)
    return np.array(locs), np.array(xs), np.array(ys), np.array(zs)

def hdf5_to_zarr(hdf5_path, zarr_path):
    with NWBHDF5IO(hdf5_path, mode='r') as read_io:  # Create HDF5 IO object for read
        with NWBZarrIO(str(zarr_path), 'w') as export_io:  # Create Zarr IO object for write
            export_io.export(src_io=read_io, write_args=dict(link_data=False))  # Export from HDF5 to Zarr
    print(f'zarr file made: {zarr_path}')
    # shutil.rmtree(hdf5_path)

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_ccf", type=str, default='false')
    parser.add_argument("--input_nwb_dir", type=str, default=f'nwb')
    parser.add_argument("--input_csv_dir", type=str, default='ccf')
    args = parser.parse_args()
    skip_ccf = args.skip_ccf == 'true'
    input_nwb_dir = data_folder / Path(args.input_nwb_dir)
    input_csv_dir = data_folder / Path(args.input_csv_dir)

    print('INPUT NWB DIR', input_nwb_dir)
    nwb_files = [p for p in input_nwb_dir.iterdir() if p.name.endswith(".nwb") or p.name.endswith(".nwb.zarr")]
    assert len(nwb_files) == 1, f"Attach one base NWB file at a time. {len(nwb_files)} found"
    input_nwb_path = nwb_files[0]
    print('INPUT NWB', input_nwb_path)

    # clear scratch dir if not empty
    print(f'Emptying scratch dir {scratch_folder}')
    for scratch_file in scratch_folder.iterdir():
        shutil.rmtree(scratch_file)    

    # determine if file is zarr or hdf5, and copy it to results
    scratch_nwb_path = scratch_folder / input_nwb_path.name
    result_nwb_path = results_folder / input_nwb_path.name
    copy_to = result_nwb_path if skip_ccf else scratch_nwb_path
    print(f"copying working files from {input_nwb_path} to {copy_to}")
    if input_nwb_path.is_dir():
        assert (input_nwb_path / ".zattrs").is_file(), f"{input_nwb_path.name} is not a valid Zarr folder"
        NWB_BACKEND = "zarr"
        io_class = NWBZarrIO
        shutil.copytree(input_nwb_path, copy_to, dirs_exist_ok=True)
    else:
        NWB_BACKEND = "hdf5"
        io_class = NWBHDF5IO
        shutil.copyfile(input_nwb_path, copy_to)
    #    NWB_BACKEND = "zarr"
    #    io_class = NWBZarrIO
    #    hdf5_to_zarr(input_nwb_path, scratch_nwb_path)

    print(f"NWB backend: {NWB_BACKEND}")
    if skip_ccf:
        print('Skipping addition of CCF, outputting NWB file as-is')
        return

    probe_csvs = [p for p in input_csv_dir.iterdir() if p.name.endswith('sorted_ccf_regions.csv')]
    assert len(probe_csvs) > 0, f'No CCF CSVs found to use. If CCF addition should be skipped, use `--skip_cff True`'

    print('Building CCF Map from .CSVs')
    ccf_map = build_ccf_map(probe_csvs)

    print('Reading NWB in append mode:', scratch_nwb_path)
    print(os.listdir(scratch_folder))
    if scratch_nwb_path.is_dir():
        print(os.listdir(scratch_nwb_path))
    with io_class(str(scratch_nwb_path), mode='a') as read_io:
        nwb = read_io.read()

        print('Getting new electrode columns')
        locs, xs, ys, zs = get_new_electrode_colums(nwb, ccf_map)

        nwb.electrodes.location.data[:] = np.array(locs)
        nwb.electrodes.add_column('x', 'ccf x coordinate', data=xs)
        nwb.electrodes.add_column('y', 'ccf y coordinate', data=ys)
        nwb.electrodes.add_column('z', 'ccf z coordinate', data=zs)

        print('Exporting to NWB:',result_nwb_path)
        with io_class(str(result_nwb_path), "w") as export_io:
            export_io.export(src_io=read_io, nwbfile=nwb, write_args={'link_data': False})
        print(f"Done writing {result_nwb_path}")


if __name__ == "__main__": run()