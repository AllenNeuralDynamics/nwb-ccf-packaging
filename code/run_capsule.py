"""top level run script"""

import argparse
from pathlib import Path
from hdmf_zarr.nwb import NWBZarrIO
from pynwb import NWBHDF5IO
import csv
import json
import os
import shutil
from pathlib import Path
import re
import ibllib.atlas as atlas
from typing import Union

import numpy as np
from hdmf.common.table import DynamicTable
from hdmf_zarr.nwb import NWBZarrIO
from pynwb import NWBHDF5IO
import pandas as pd

import ibllib.atlas as atlas
import SimpleITK as sitk

data_folder = Path("/data/")
scratch_folder = Path("/scratch/")
results_folder = Path("/results/")

RESOLUTION_UM = 25

def _verify_point(point: int, bound: int) -> None:
    """
    Verify that a point index is within the valid bounds.

    Parameters
    ----------
    point : int
        The point index to verify (e.g., voxel index along one axis).

    bound : int
        The exclusive upper bound (e.g., size of the axis). The valid range is [0, bound).

    Raises
    ------
    ValueError
        If the point is negative or greater than or equal to the bound.
    """
    if point < 0 or point >= bound:
        raise ValueError(f"Value {point} does not fall within range of 0 to {bound}")

def convert_ibl_bregma_to_ccf_microns(brain_atlas: atlas.AllenAtlas, ccf_volume: sitk.Image, ccf_array: np.ndarray, point: tuple[int, int, int]) -> tuple[int, int, int]:
    """
    Convert a coordinate from IBL Bregma space to Common Coordinate Framework (CCF) space.

    Parameters
    ----------
    brain_atlas : atlas.AllenAtlas
        An instance of the AllenAtlas from the ibllib.atlas
    
    ccf_volume : sitk.Image
        A SimpleITK image representing the CCF volume, used for coordinate transformation.
    
    ccf_array: np.ndarray
        Array represent the CCF volume. Shape (ml, dv, ap)

    point : tuple of int
        A 3D point (x, y, z) in IBL Bregma space to be converted to CCF coordinates.

    Returns
    -------
    tuple of int
        The corresponding 3D point (x, y, z) in CCF space in microns.
        x - AP
        y - DV
        z - ML
    """
    ccf_point = brain_atlas.xyz2ccf(np.array(point), ccf_order='mlapdv', mode='wrap')
    ccf_point_indices = np.array(ccf_volume.TransformPhysicalPointToIndex(ccf_point / 1e3)) # returns point in order AP-DV-ML
    ccf_point_indices[2] = -ccf_point_indices[2]
    ccf_point_indices[1] = -ccf_point_indices[1]

    # check points are within bounds 
    _verify_point(ccf_point_indices[0], ccf_array.shape[2]) # AP
    _verify_point(ccf_point_indices[1], ccf_array.shape[1]) # DV
    _verify_point(ccf_point_indices[2], ccf_array.shape[0]) # ML

    return ccf_point_indices * RESOLUTION_UM


def extract_session_name_from_nwb(nwb_path):
    match = re.match(r"(ecephys_\d+_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", nwb_path.name)
    if match:
        return match.group(1)
    return None


# converts channel name of the form 'LFP1' or 'AP1' to idx of the form 0
def channel_name_to_idx(channel_name):
    """
    Convert a channel name to a corresponding channel index.

    This function processes a channel name by removing specific substrings ('LFP' and 'AP')
    and then attempts to convert the resulting string into an integer. The function returns
    the integer index of the channel, adjusting by subtracting 1. If the conversion fails,
    an exception is raised indicating an unexpected channel name format.

    Parameters
    ----------
    channel_name : str
        The name of the channel as a string. The function expects the name to possibly
        include the substrings 'LFP' or 'AP', which will be removed during processing.

    Returns
    -------
    int
        The channel index, derived from the numerical part of the channel name, with
        1 subtracted from it.
    """
    stripped_name = channel_name.replace("LFP", "").replace("AP", "")
    try:
        return int(stripped_name) - 1
    except:
        raise Exception("Unexpected channel name format")



def build_ccf_map(ccf_json_files, ccf_volume: sitk.Image, brain_atlas: Union[atlas.AllenAtlas, None] = None) -> dict:
    """
    Build a CCF (Common Coordinate Framework) map from a list of JSON files.

    This function processes multiple JSON files that describe the location and
    structure of brain regions for different probes. It extracts the relevant
    information from each JSON file and constructs a dictionary mapping each probe
    and channel to the corresponding brain region and coordinates (x, y, z).

    Parameters
    ----------
    ccf_json_files : list of pathlib.Path
        A list of file paths (as Path objects) to the JSON files containing CCF data.
        Each file is expected to have data for various channels, with information
        on the brain region and the corresponding coordinates.

    ccf_volume : sitk.Image
        A SimpleITK image representing the CCF volume, used for coordinate transformation.

    brain_atlas: Union[atlas.AllenAtlas, None]. Default None
        An instance of the AllenAtlas from the ibllib.atlas

    Returns
    -------
    dict
        A dictionary where the keys are tuples of the form `(probe_name, channel_id)`,
        and the values are lists containing the structure name and coordinates `[structure, x, y, z]`.
    """
    ccf_map = {}
    ccf_array = sitk.GetArrayFromImage(ccf_volume)

    for ccf_json_path in ccf_json_files:
        print("Reading", ccf_json_path)
        probe_name = ccf_json_path.parent.stem
        probe_name = probe_name[0].lower() + probe_name[1:]
        with open(ccf_json_path, "r") as f:
            ccf_json_info = json.load(f)

        for channel in ccf_json_info:
            try:
                channel_id = channel[channel.index("_") + 1 :]
            except:
                print(f'skipping channel {channel}')    
                continue
            structure = ccf_json_info[channel]["brain_region"]
            x = ccf_json_info[channel]["x"]
            y = ccf_json_info[channel]["y"]
            z = ccf_json_info[channel]["z"]

            if brain_atlas is not None:
                ccf_point_microns = convert_ibl_bregma_to_ccf_microns(brain_atlas, ccf_volume, ccf_array (x, y, z))
            else:
                ccf_point_indices = np.array(ccf_volume.TransformPhysicalPointToIndex(np.array(x, y, z)))
                # check points are within bounds 
                _verify_point(ccf_point_indices[0], ccf_array.shape[2]) # AP
                _verify_point(ccf_point_indices[1], ccf_array.shape[1]) # DV
                _verify_point(ccf_point_indices[2], ccf_array.shape[0]) # ML
                ccf_point_microns = ccf_point_indices * RESOLUTION_UM
            
            ccf_map[probe_name, int(channel_id)] = [structure, ccf_point_microns[0], ccf_point_microns[1], ccf_point_microns[2]]

    return ccf_map


def correct_isi_locations(ccf_map, area_classifications):
    print(area_classifications)
    for index, row in area_classifications.iterrows():
        probe_name = f"probe{row['Probe']}"
        corrected_area = row['Area']
        if corrected_area.lower() == 'nonvis':
            continue

        print(f'Correcting {probe_name} visual areas with {corrected_area}')
        n_corrected_areas = 0
        for electrode in ccf_map:
            electrode_region = ccf_map[electrode][0]
            if 'vis' in electrode_region.lower():
                ccf_map[electrode][0] = corrected_area
                n_corrected_areas += 1

        print(f'Corrected {n_corrected_areas} areas')

    return ccf_map

def get_isi_column(nwb, area_classifications):
    """
    """
    print(area_classifications)
    isi_locs = []
    print("electrode columns:",len(nwb.electrodes))
    for row in nwb.electrodes:
        probe_name = row["group_name"].item()
        probe_letter = probe_name[-1].upper()
        targeted_area = area_classifications.loc[area_classifications['Probe'] == probe_letter, 'Area'].iloc[0]

        isi_locs.append(targeted_area)

    assert len(isi_locs) == len(nwb.electrodes)
    return np.array(isi_locs)


def get_new_electrode_colums(nwb, ccf_map):
    """
    Extract the anatomical location (brain region) and coordinates (x, y, z)
    for each electrode from the provided NWB file and CCF map.

    This function loops over all electrodes in the given NWB file, retrieves
    the probe name and channel ID, and looks up the corresponding brain region
    and coordinates from the CCF map. It returns the lists of brain regions and
    coordinates for the electrodes.

    Parameters
    ----------
    nwb : pynwb.file.NWBFile
        An NWB (Neurodata Without Borders) file containing electrode information
        (e.g., electrode group names and channel names).

    ccf_map : dict
        A dictionary mapping probe names and channel IDs (as tuples) to the
        corresponding brain region and coordinates. Each key is a tuple of the form
        `(probe_name, channel_id)`, and each value is a list `[structure, x, y, z]`.

    Returns
    -------
    tuple of lists
        A tuple containing four lists:
        - `locs`: A list of brain regions (structure names) corresponding to each electrode.
        - `xs`: A list of x-coordinates for each electrode.
        - `ys`: A list of y-coordinates for each electrode.
        - `zs`: A list of z-coordinates for each electrode.
    """
    locs, xs, ys, zs = [], [], [], []
    print("electrode columns:",len(nwb.electrodes))
    for row in nwb.electrodes:
        probe_name = row["group_name"].item()
        probe_name = probe_name.replace(" ", "")
        probe_name = probe_name[0].lower() + probe_name[1:]
        channel_id = channel_name_to_idx(row["channel_name"].item())
        try:
            structure, x, y, z = ccf_map[probe_name, channel_id]
        except KeyError:
            raise Exception(
                f"CCF information for an electrode ({probe_name}, channel {channel_id}) not found. Perhaps no output from IBL alignment for {probe_name}"
            )

        locs.append(structure)
        xs.append(x)
        ys.append(y)
        zs.append(z)

    assert len(locs) == len(nwb.electrodes)
    return np.array(locs), np.array(xs), np.array(ys), np.array(zs)


def zarr_to_hdf5(zarr_path, output_dir):
    hdf5_path = output_dir / zarr_path.name
    with NWBZarrIO(str(zarr_path), mode='r') as read_io:  # Create Zarr IO object for read
        with NWBHDF5IO(hdf5_path, 'w') as export_io:  # Create HDF5 IO object for write
            export_io.export(src_io=read_io, write_args=dict(link_data=False))  # Export from Zarr to HDF5
    print(f'hdf5 file made: {hdf5_path}')


def empty_folder(path):
    for file in path.iterdir():
        try:
            if file.is_symlink() or file.is_file():
                file.unlink()
            else:
                shutil.rmtree(file)

        except Exception as e:
            print(f"Failed to empty {file} from folder:", path, 'error:', e)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_ccf", type=str, default='false')
    parser.add_argument("--only_regions", type=str, default='false')
    parser.add_argument("--isi_correction", type=str, default='True')
    parser.add_argument("--behavior_dir", type=str, default="session/behavior")
    parser.add_argument("--input_nwb_dir", type=str, default=f'nwb')
    # parser.add_argument("--input_csv_dir", type=str, default='ccf')
    parser.add_argument("--input_ccf_dir", type=str, default="ccf")
    parser.add_argument("--convert_ibl_bregma_to_ccf", type=str, default='false')

    args = parser.parse_args()
    skip_ccf = args.skip_ccf in ['True','true','T','t']
    only_regions = args.only_regions in ['True','true','T','t']
    isi_correction = args.isi_correction in ['True','true','T','t']
    convert_ibl_bregma_to_ccf = args.convert_ibl_bregma_to_ccf in ['True','true','T','t']

    behavior_dir = data_folder / Path(args.behavior_dir)
    input_nwb_dir = data_folder / Path(args.input_nwb_dir)
    input_ccf_dir = data_folder / Path(args.input_ccf_dir)

    print('INPUT NWB DIR', input_nwb_dir)
    nwb_files = [p for p in input_nwb_dir.iterdir() if p.name.endswith(".nwb") or p.name.endswith(".nwb.zarr")]
    assert len(nwb_files) == 1, f"Attach one base NWB file at a time. {len(nwb_files)} found"
    input_nwb_path = nwb_files[0]
    print('INPUT NWB', input_nwb_path)

    # clear scratch dir if not empty
    print(f'Emptying scratch dir {scratch_folder}')
    empty_folder(scratch_folder)

    # clear results dir if not empty
    print(f'Emptying results dir {results_folder}')
    empty_folder(results_folder)

    # determine if file is zarr or hdf5, and copy it to results
    scratch_nwb_path = scratch_folder / input_nwb_path.name
    result_nwb_path = results_folder / input_nwb_path.name
    copied_nwb_path = result_nwb_path if skip_ccf else scratch_nwb_path
    print(f"copying working files from {input_nwb_path} to {copied_nwb_path}")
    if input_nwb_path.is_dir():
        # NWB_BACKEND = "hdf5"
        # io_class = NWBHDF5IO
        # zarr_to_hdf5(input_nwb_path, copied_nwb_path.parent)
        assert (input_nwb_path / ".zattrs").is_file(), f"{input_nwb_path.name} is not a valid Zarr folder"
        NWB_BACKEND = "zarr"
        io_class = NWBZarrIO
        shutil.copytree(input_nwb_path, copied_nwb_path, dirs_exist_ok=True)
    else:
        NWB_BACKEND = "hdf5"
        io_class = NWBHDF5IO
        shutil.copyfile(input_nwb_path, copied_nwb_path)

    print(f"NWB backend: {NWB_BACKEND}")
    if skip_ccf:
        print('Skipping addition of CCF, outputting NWB file as-is')
        return

    session_id = extract_session_name_from_nwb(input_nwb_path)
    print(f'Looking for CCF for session_id {session_id}')
    ccf_json_files = tuple(input_ccf_dir.rglob(f"**/{session_id}/**/Probe*/*locations*.json"))
    if not ccf_json_files:
        raise FileNotFoundError("No IBL jsons attached")
    print(f"Found {len(ccf_json_files)} ccf jsons")

    # probe_csvs = [p for p in input_csv_dir.iterdir() if p.name.endswith('sorted_ccf_regions.csv')]
    # assert len(probe_csvs) > 0, f'No CCF CSVs found to use. If CCF addition should be skipped, use `--skip_cff True`'

    brain_atlas = None
    ccf_volume = sitk.ReadImage('/root/capsule/data/allen_mouse_ccf/annotation/ccf_2017/annotation_25.nii.gz')
    print("Starting to add to NWB. Coordinates will be in microns")
    if convert_ibl_bregma_to_ccf:
        print("Alignment was done in IBL bregma space. Will be converting back to CCF")
        # resolution of Allen CCF atlas
        brain_atlas = atlas.AllenAtlas(RESOLUTION_UM)

    print("Building CCF Map from IBL jsons")
    print(repr(only_regions))
    ccf_map = build_ccf_map(ccf_json_files, ccf_volume, brain_atlas=brain_atlas)

    print("Reading NWB in append mode:", result_nwb_path)
    print(repr(only_regions))
    with io_class(str(copied_nwb_path), mode="a") as read_io:
        nwb = read_io.read()

        print("Getting new electrode columns")
        print(repr(only_regions))
        locs, xs, ys, zs = get_new_electrode_colums(nwb, ccf_map)
        if len(locs) < len(nwb.electrodes):
            for i in range(len(nwb.electrodes) - len(locs)):
                locs.append("unknown")
                xs.append(np.nan)
                ys.append(np.nan)
                zs.append(np.nan)

        nwb.electrodes.location.data[:] = np.array(locs)
        if not only_regions:
            nwb.electrodes.add_column("x", "ccf x coordinate", data=xs)
            nwb.electrodes.add_column("y", "ccf y coordinate", data=ys)
            nwb.electrodes.add_column("z", "ccf z coordinate", data=zs)

        if isi_correction:
            area_classifications = pd.read_csv(next(behavior_dir.rglob('*areaClassifications.csv')))
            isi_column = get_isi_column(nwb, area_classifications)
            nwb.electrodes.add_column("isi_region", "ISI mapped targeted location", data=isi_column)

        print("at end, electrodes table has len",len(nwb.electrodes))
        print('Exporting to NWB:',result_nwb_path)
        with io_class(str(result_nwb_path), "w") as export_io:
            export_io.export(src_io=read_io, nwbfile=nwb, write_args={'link_data': False})
        print(f"Done writing {result_nwb_path}")


if __name__ == "__main__":
    run()
